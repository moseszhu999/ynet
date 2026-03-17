//! Core P2P network node using libp2p.

use futures::StreamExt;
use libp2p::{
    gossipsub, identify, kad, mdns, noise,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, Multiaddr, PeerId, SwarmBuilder,
};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::mpsc;

const PROTOCOL_VERSION: &str = "/ynet/0.1.0";
const GOSSIP_TOPIC: &str = "ynet-global";

#[derive(NetworkBehaviour)]
struct YnetBehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
    kademlia: kad::Behaviour<kad::store::MemoryStore>,
    identify: identify::Behaviour,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub peer_id: String,
    pub addresses: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatus {
    pub node_id: String,
    pub connected_peers: usize,
    pub is_online: bool,
    pub listening_addresses: Vec<String>,
    pub peers: Vec<PeerInfo>,
}

#[derive(Debug, Clone)]
pub enum NetworkEvent {
    PeerConnected(String),
    PeerDisconnected(String),
    MessageReceived {
        from: String,
        topic: String,
        data: Vec<u8>,
    },
}

/// Command sent to the network event loop.
enum NetworkCommand {
    GetStatus(tokio::sync::oneshot::Sender<NetworkStatus>),
    Broadcast { topic: String, data: Vec<u8> },
    Dial(Multiaddr),
}

/// Handle to interact with the running P2P network.
pub struct NetworkNode {
    command_tx: mpsc::Sender<NetworkCommand>,
    event_rx: mpsc::Receiver<NetworkEvent>,
    local_peer_id: String,
}

impl NetworkNode {
    /// Start the P2P network node. Returns a handle for sending commands and receiving events.
    pub async fn start(listen_port: u16, seed_nodes: Vec<String>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let (command_tx, command_rx) = mpsc::channel(256);
        let (event_tx, event_rx) = mpsc::channel(256);

        // Build swarm
        let mut swarm = SwarmBuilder::with_new_identity()
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_behaviour(|key| {
                // GossipSub
                let gossipsub_config = gossipsub::ConfigBuilder::default()
                    .heartbeat_interval(Duration::from_secs(5))
                    .validation_mode(gossipsub::ValidationMode::Strict)
                    .build()
                    .expect("valid gossipsub config");
                let gossipsub = gossipsub::Behaviour::new(
                    gossipsub::MessageAuthenticity::Signed(key.clone()),
                    gossipsub_config,
                )
                .expect("valid gossipsub behaviour");

                // mDNS for LAN discovery
                let mdns = mdns::tokio::Behaviour::new(
                    mdns::Config::default(),
                    key.public().to_peer_id(),
                )
                .expect("valid mdns");

                // Kademlia DHT for WAN discovery
                let mut kademlia = kad::Behaviour::new(
                    key.public().to_peer_id(),
                    kad::store::MemoryStore::new(key.public().to_peer_id()),
                );
                kademlia.set_mode(Some(kad::Mode::Server));

                // Identify protocol
                let identify = identify::Behaviour::new(identify::Config::new(
                    PROTOCOL_VERSION.to_string(),
                    key.public(),
                ));

                YnetBehaviour {
                    gossipsub,
                    mdns,
                    kademlia,
                    identify,
                }
            })?
            .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
            .build();

        let local_peer_id = *swarm.local_peer_id();
        let local_peer_id_str = local_peer_id.to_string();

        // Subscribe to gossip topics
        let topic = gossipsub::IdentTopic::new(GOSSIP_TOPIC);
        swarm.behaviour_mut().gossipsub.subscribe(&topic)?;
        let chain_topic = gossipsub::IdentTopic::new("ynet-chain");
        swarm.behaviour_mut().gossipsub.subscribe(&chain_topic)?;
        let tasks_topic = gossipsub::IdentTopic::new("ynet-tasks");
        swarm.behaviour_mut().gossipsub.subscribe(&tasks_topic)?;
        let inference_topic = gossipsub::IdentTopic::new("ynet-inference");
        swarm.behaviour_mut().gossipsub.subscribe(&inference_topic)?;

        // Listen on the specified port
        let listen_addr: Multiaddr = format!("/ip4/0.0.0.0/tcp/{listen_port}").parse()?;
        swarm.listen_on(listen_addr)?;

        info!("P2P node started: {local_peer_id_str}");

        // Dial seed nodes
        for addr_str in &seed_nodes {
            if let Ok(addr) = addr_str.parse::<Multiaddr>() {
                info!("Dialing seed node: {addr}");
                let _ = swarm.dial(addr);
            } else {
                warn!("Invalid seed node address: {addr_str}");
            }
        }

        // Spawn the event loop
        let peer_id_for_loop = local_peer_id_str.clone();
        tokio::spawn(event_loop(swarm, command_rx, event_tx, peer_id_for_loop));

        Ok(NetworkNode {
            command_tx,
            event_rx,
            local_peer_id: local_peer_id_str,
        })
    }

    pub fn local_peer_id(&self) -> &str {
        &self.local_peer_id
    }

    pub async fn get_status(&self) -> NetworkStatus {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.command_tx.send(NetworkCommand::GetStatus(tx)).await;
        rx.await.unwrap_or(NetworkStatus {
            node_id: self.local_peer_id.clone(),
            connected_peers: 0,
            is_online: false,
            listening_addresses: vec![],
            peers: vec![],
        })
    }

    pub async fn broadcast(&self, topic: &str, data: Vec<u8>) {
        let _ = self
            .command_tx
            .send(NetworkCommand::Broadcast {
                topic: topic.to_string(),
                data,
            })
            .await;
    }

    pub async fn dial(&self, addr: &str) {
        if let Ok(multiaddr) = addr.parse() {
            let _ = self.command_tx.send(NetworkCommand::Dial(multiaddr)).await;
        }
    }

    pub async fn recv_event(&mut self) -> Option<NetworkEvent> {
        self.event_rx.recv().await
    }
}

async fn event_loop(
    mut swarm: libp2p::Swarm<YnetBehaviour>,
    mut command_rx: mpsc::Receiver<NetworkCommand>,
    event_tx: mpsc::Sender<NetworkEvent>,
    local_peer_id: String,
) {
    let mut connected_peers: HashMap<PeerId, Vec<Multiaddr>> = HashMap::new();
    let mut listening_addresses: Vec<Multiaddr> = Vec::new();

    loop {
        tokio::select! {
            // Handle commands from the application
            cmd = command_rx.recv() => {
                match cmd {
                    Some(NetworkCommand::GetStatus(reply)) => {
                        let peers: Vec<PeerInfo> = connected_peers
                            .iter()
                            .map(|(id, addrs)| PeerInfo {
                                peer_id: id.to_string(),
                                addresses: addrs.iter().map(|a| a.to_string()).collect(),
                            })
                            .collect();
                        let _ = reply.send(NetworkStatus {
                            node_id: local_peer_id.clone(),
                            connected_peers: connected_peers.len(),
                            is_online: true,
                            listening_addresses: listening_addresses.iter().map(|a| a.to_string()).collect(),
                            peers,
                        });
                    }
                    Some(NetworkCommand::Broadcast { topic, data }) => {
                        let gossip_topic = gossipsub::IdentTopic::new(&topic);
                        if let Err(e) = swarm.behaviour_mut().gossipsub.publish(gossip_topic, data) {
                            warn!("Failed to publish message: {e}");
                        }
                    }
                    Some(NetworkCommand::Dial(addr)) => {
                        info!("Dialing: {addr}");
                        let _ = swarm.dial(addr);
                    }
                    None => break,
                }
            }

            // Handle swarm events
            event = swarm.select_next_some() => {
                match event {
                    SwarmEvent::NewListenAddr { address, .. } => {
                        info!("Listening on: {address}");
                        listening_addresses.push(address);
                    }
                    SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                        info!("Connected to: {peer_id}");
                        connected_peers.entry(peer_id).or_default();
                        let _ = event_tx.send(NetworkEvent::PeerConnected(peer_id.to_string())).await;
                    }
                    SwarmEvent::ConnectionClosed { peer_id, .. } => {
                        info!("Disconnected from: {peer_id}");
                        connected_peers.remove(&peer_id);
                        let _ = event_tx.send(NetworkEvent::PeerDisconnected(peer_id.to_string())).await;
                    }

                    // mDNS events — auto-discover LAN peers
                    SwarmEvent::Behaviour(YnetBehaviourEvent::Mdns(mdns::Event::Discovered(list))) => {
                        for (peer_id, addr) in list {
                            info!("mDNS discovered: {peer_id} at {addr}");
                            swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                            swarm.behaviour_mut().kademlia.add_address(&peer_id, addr.clone());
                            let _ = swarm.dial(addr);
                        }
                    }
                    SwarmEvent::Behaviour(YnetBehaviourEvent::Mdns(mdns::Event::Expired(list))) => {
                        for (peer_id, _) in list {
                            debug!("mDNS expired: {peer_id}");
                        }
                    }

                    // GossipSub messages
                    SwarmEvent::Behaviour(YnetBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                        propagation_source,
                        message,
                        ..
                    })) => {
                        let topic = message.topic.to_string();
                        info!("GossipSub message from {propagation_source} on topic {topic}");
                        let _ = event_tx.send(NetworkEvent::MessageReceived {
                            from: propagation_source.to_string(),
                            topic,
                            data: message.data,
                        }).await;
                    }

                    // Identify events — learn peer addresses for Kademlia
                    SwarmEvent::Behaviour(YnetBehaviourEvent::Identify(identify::Event::Received {
                        peer_id,
                        info: identify::Info { listen_addrs, .. },
                        ..
                    })) => {
                        for addr in &listen_addrs {
                            swarm.behaviour_mut().kademlia.add_address(&peer_id, addr.clone());
                        }
                        connected_peers.insert(peer_id, listen_addrs);
                    }

                    _ => {}
                }
            }
        }
    }
}
