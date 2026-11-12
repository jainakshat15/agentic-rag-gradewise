---
title: "CS201 Unit 2: Computer Networks"
subject: Computer Networks
unit: 2
---

# Unit 2: Computer Networks

**CS201 Systems Fundamentals** · Northbridge Institute, School of Computing
Course notes by Dr. Emma Carter · Edition 2026

## Learning outcomes

By the end of this unit you should be able to:

1. Describe the layered architecture of networks using the OSI and TCP/IP models.
2. Explain how data is encapsulated as it travels down and up the protocol stack.
3. Work with IPv4 addresses, subnet masks and CIDR, and calculate hosts per subnet.
4. Compare TCP and UDP, and describe the TCP three-way handshake.
5. Explain how DNS resolves a name and how an HTTP request is served.
6. Calculate transmission and propagation delays.

## 2.1 What is a network?

A **computer network** is a collection of devices (**hosts**) connected so that they can exchange data. Networks are classified by scale: a **LAN** (Local Area Network) covers a building or campus; a **WAN** (Wide Area Network) spans cities or countries. The **Internet** is a network of networks.

There are two fundamental ways to move data across a network:

- **Circuit switching:** a dedicated path is reserved for the whole conversation, as in the traditional telephone network. Performance is guaranteed, but capacity is wasted when nobody is talking.
- **Packet switching:** data is split into **packets** that travel independently and share links with other traffic. Links are used efficiently, but packets can be delayed, reordered or lost. The Internet is packet-switched.

## 2.2 Layered models

Networking is complex, so it is divided into **layers**. Each layer provides services to the layer above and uses the services of the layer below. Layers can be changed independently: Wi-Fi can replace Ethernet without changing the web browser.

### The OSI reference model (7 layers)

| # | Layer | Responsibility | Example |
|---|---|---|---|
| 7 | Application | Network services for applications | HTTP, DNS, SMTP |
| 6 | Presentation | Data format, encryption, compression | TLS encryption, character encoding |
| 5 | Session | Setting up and managing sessions | Session resumption |
| 4 | Transport | End-to-end delivery between processes | TCP, UDP |
| 3 | Network | Logical addressing and routing between networks | IP |
| 2 | Data link | Delivery between neighbouring nodes on one link | Ethernet, Wi-Fi (MAC addresses) |
| 1 | Physical | Transmitting raw bits over a medium | Cables, radio signals |

### The TCP/IP model (4 layers)

The model the Internet actually uses has four layers: **Application** (OSI 5–7), **Transport** (OSI 4), **Internet** (OSI 3) and **Link** (OSI 1–2).

### Encapsulation

As data moves **down** the stack on the sender, each layer adds its own **header**: the transport layer creates a **segment**, the network layer wraps it in a **packet** (datagram), and the link layer wraps that in a **frame**. On the receiver, each layer removes its header as the data moves **up** the stack. This process is called **encapsulation** and **decapsulation**.

## 2.3 The link layer

Every network interface has a **MAC address**: a 48-bit hardware address, usually written as six hexadecimal pairs (for example `3C:22:FB:9A:10:7E`). MAC addresses identify devices *on the same local network*.

A **switch** forwards frames within a LAN using MAC addresses. A **router** connects different networks and forwards packets using IP addresses.

When a host knows a neighbour's IP address but not its MAC address, it uses the **Address Resolution Protocol (ARP)**: it broadcasts "who has IP 192.168.1.20?" and the owner replies with its MAC address.

## 2.4 The network layer: IP addressing

An **IPv4 address** is 32 bits, written as four decimal numbers (octets) separated by dots, for example `192.168.10.77`. Each address has two parts: a **network** part identifying the network, and a **host** part identifying the device on that network.

### Subnet masks and CIDR

A **subnet mask** marks which bits belong to the network part. In **CIDR notation**, the mask is written as the number of network bits after a slash: `/24` means the first 24 bits are the network part (mask `255.255.255.0`).

For a prefix of length n:

- the number of host bits is 32 − n;
- the total number of addresses in the subnet is 2^(32 − n);
- the number of **usable host addresses** is 2^(32 − n) − 2.

We subtract 2 because two addresses are reserved: the **network address** (all host bits 0) identifies the subnet itself, and the **broadcast address** (all host bits 1) sends to every host on the subnet.

### Worked example: a /26 subnet

A host has address **192.168.10.77/26**.

1. Host bits = 32 − 26 = **6**, so the subnet has 2^6 = **64 addresses**.
2. Usable hosts = 64 − 2 = **62**.
3. Subnets of size 64 start at multiples of 64 in the last octet: 0, 64, 128, 192. Since 77 lies between 64 and 127, the **network address is 192.168.10.64**.
4. The **broadcast address** is the last address in the block: **192.168.10.127**.
5. Usable host range: **192.168.10.65 to 192.168.10.126**.
6. The subnet mask for /26 is **255.255.255.192**.

### Private addresses and NAT

Some ranges are **private**, reserved for use inside organisations and not routed on the public Internet: `10.0.0.0/8`, `172.16.0.0/12` and `192.168.0.0/16`. **Network Address Translation (NAT)**, usually done by a home router, lets many devices with private addresses share one public IP address.

### Routing and IPv6

**Routers** forward packets hop by hop, using a **routing table** to choose the next hop for each destination network. IPv4's 32-bit space (about 4.3 billion addresses) has run out; **IPv6** uses **128-bit** addresses, written in hexadecimal, for example `2001:db8::1`.

## 2.5 The transport layer: TCP and UDP

IP delivers packets *between hosts*. The transport layer delivers data *between processes* on those hosts, identified by **port numbers** (for example, HTTPS uses port 443 and DNS uses port 53).

| | TCP | UDP |
|---|---|---|
| Connection | Connection-oriented (handshake first) | Connectionless |
| Reliability | Reliable: lost data is retransmitted | Unreliable: no retransmission |
| Ordering | Delivers bytes in order | No ordering guarantee |
| Flow and congestion control | Yes | No |
| Header size | 20 bytes minimum | 8 bytes |
| Speed / overhead | Higher overhead | Lower overhead, lower latency |
| Typical uses | Web (HTTP), email, file transfer | DNS queries, video calls, online games, streaming |

### The TCP three-way handshake

Before sending data, TCP establishes a connection in three steps:

1. **SYN:** the client sends a segment with the SYN flag and its initial sequence number.
2. **SYN-ACK:** the server replies with SYN and ACK flags, acknowledging the client's sequence number and sending its own.
3. **ACK:** the client acknowledges the server's sequence number. The connection is now established.

### How TCP achieves reliability

- **Sequence numbers** label every byte so the receiver can reorder data and detect gaps.
- **Acknowledgements (ACKs)** tell the sender what has arrived.
- **Retransmission** resends data that is not acknowledged before a timeout.
- **Flow control** (a sliding window advertised by the receiver) stops a fast sender overwhelming a slow *receiver*.
- **Congestion control** (for example slow start, where the sending window grows exponentially until loss is detected) stops senders overwhelming the *network*.

### Common misconceptions

- *"UDP is bad because it is unreliable."* For real-time applications, a late packet is useless; UDP avoids the delay of retransmissions, and the application can handle loss itself.
- *"Flow control and congestion control are the same."* Flow control protects the receiver; congestion control protects the network.

## 2.6 The application layer

### DNS: from names to addresses

People use names like `www.northbridge.edu`; IP needs addresses. The **Domain Name System (DNS)** is a distributed, hierarchical database that translates names into IP addresses. Resolving a name typically goes like this:

1. The browser checks its own cache, then asks the operating system's resolver.
2. If the answer is not cached, the query goes to a **recursive resolver** (often run by the ISP).
3. The recursive resolver asks a **root server**, which refers it to the **top-level domain (TLD)** server for `.edu`.
4. The TLD server refers it to the **authoritative name server** for `northbridge.edu`.
5. The authoritative server returns the IP address, which is cached at each level for its **time to live (TTL)**.

DNS normally uses **UDP on port 53** because queries are small and fast.

### HTTP and HTTPS

**HTTP** is a request–response protocol. A client sends a request with a **method** (GET retrieves a resource, POST submits data, PUT replaces, DELETE removes) and the server replies with a **status code**:

| Code range | Meaning | Example |
|---|---|---|
| 2xx | Success | 200 OK |
| 3xx | Redirection | 301 Moved Permanently |
| 4xx | Client error | 404 Not Found |
| 5xx | Server error | 500 Internal Server Error |

HTTP is **stateless**: each request is independent, so applications use cookies or tokens to remember users. **HTTPS** is HTTP over **TLS**, which encrypts the traffic and uses certificates to prove the server's identity.

## 2.7 Network performance

- **Bandwidth:** the maximum data rate of a link, in bits per second.
- **Throughput:** the data rate actually achieved.
- **Latency:** the time for data to travel from source to destination.

Latency has several components. The two most important for calculations are:

- **Transmission delay** = packet size ÷ bandwidth (the time to push all the bits onto the link);
- **Propagation delay** = distance ÷ propagation speed (the time for a bit to travel along the link).

### Worked example: delays

A **1 MB** file (8 × 10^6 bits) is sent over a **100 Mbps** link of length **2,000 km**, with a propagation speed of **2 × 10^8 m/s**.

- Transmission delay = 8 × 10^6 bits ÷ 10^8 bits/s = **0.08 s = 80 ms**.
- Propagation delay = 2 × 10^6 m ÷ (2 × 10^8 m/s) = **0.01 s = 10 ms**.
- Total ≈ **90 ms** (ignoring queuing and processing delays).

Note the units: network rates use bits, and 1 byte = 8 bits.

## Key terms

| Term | Definition |
|---|---|
| Packet switching | Sending data as independent packets over shared links |
| Encapsulation | Each layer wrapping the data from the layer above in its own header |
| MAC address | A 48-bit hardware address used on the local network |
| IP address | A logical address (32-bit in IPv4) used for routing between networks |
| Subnet mask / CIDR | Marks the network part of an IP address, e.g. /26 |
| Router | A device that forwards packets between networks |
| Port | A number identifying a process on a host |
| TCP | Reliable, ordered, connection-oriented transport |
| UDP | Lightweight, connectionless, unreliable transport |
| Three-way handshake | SYN, SYN-ACK, ACK: how TCP opens a connection |
| DNS | The system that translates domain names into IP addresses |
| Latency | The time for data to travel from sender to receiver |

## Unit summary

- Networks are built in layers; the TCP/IP model has Application, Transport, Internet and Link layers.
- Encapsulation adds a header at each layer: segment, then packet, then frame.
- A /n subnet has 2^(32−n) addresses and 2^(32−n) − 2 usable hosts.
- TCP is reliable and ordered (handshake, sequence numbers, ACKs, retransmission, flow and congestion control); UDP is lightweight and fast.
- DNS resolves names through root, TLD and authoritative servers, with caching at every step.
- Total delay includes transmission delay (size ÷ bandwidth) and propagation delay (distance ÷ speed).
