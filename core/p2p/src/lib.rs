//! yNet P2P networking module
//! Built on libp2p: mDNS (LAN discovery) + Kademlia (WAN DHT) + GossipSub (broadcast).

mod network;

pub use network::{NetworkEvent, NetworkNode, NetworkStatus, PeerInfo};
