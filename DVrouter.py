"""Distributed Bellman-Ford routing with bounded metrics and poison reverse."""

from json import dumps, loads

from packet import Packet
from router import Router


class DVrouter(Router):
    # The assignment permits 16 as infinity; usable routes must cost less than 16.
    INFINITY = 16

    def __init__(self, addr, heartbeat_time):
        Router.__init__(self, addr)
        self.heartbeat_time = heartbeat_time
        self.last_time = 0
        self.adjacencies = {}  # port -> (neighbor address, outgoing cost)
        self.received_vectors = {}  # port -> (advertisement number, distances)
        self.forwarding = {}  # destination -> (total cost, outgoing port)
        self.advertisement_number = 0

    def _recompute(self):
        """Rebuild from current inputs, including increases and withdrawals."""
        candidates = {}

        def consider(destination, distance, port):
            if destination == self.addr or distance >= self.INFINITY:
                return
            # A fixed neighbor/port ordering makes equal-cost choices reproducible.
            rank = (distance, self.adjacencies[port][0], port)
            if destination not in candidates or rank < candidates[destination]:
                candidates[destination] = rank

        for port, (neighbor, link_cost) in self.adjacencies.items():
            consider(neighbor, link_cost, port)
            _, vector = self.received_vectors.get(port, (-1, {}))
            for destination, distance in vector.items():
                consider(destination, link_cost + distance, port)

        replacement = {
            destination: (rank[0], rank[2])
            for destination, rank in candidates.items()
        }
        changed = replacement != self.forwarding
        self.forwarding = replacement
        return changed

    def _advertise(self):
        """Send our own vector, tailored to each neighbor, never a received one."""
        self.advertisement_number += 1
        for port, (neighbor, _) in self.adjacencies.items():
            vector = {self.addr: 0}
            for destination, (distance, next_port) in self.forwarding.items():
                next_neighbor = self.adjacencies[next_port][0]
                vector[destination] = (
                    self.INFINITY if next_neighbor == neighbor else distance
                )
            content = dumps({"number": self.advertisement_number, "vector": vector})
            self.send(port, Packet(Packet.ROUTING, self.addr, neighbor, content))

    def handle_packet(self, port, packet):
        if packet.is_traceroute:
            entry = self.forwarding.get(packet.dst_addr)
            if entry is not None:
                self.send(entry[1], packet)
            return
        if not packet.is_routing or port not in self.adjacencies:
            return
        if packet.src_addr != self.adjacencies[port][0]:
            return

        update = loads(packet.content)
        previous_number, _ = self.received_vectors.get(port, (-1, {}))
        if update["number"] <= previous_number:
            return  # Link latency may deliver older advertisements after newer ones.
        # Replace the complete vector: an omitted destination has been withdrawn.
        self.received_vectors[port] = (update["number"], update["vector"])
        if self._recompute():
            self._advertise()

    def handle_new_link(self, port, endpoint, cost):
        self.adjacencies[port] = (endpoint, cost)
        self.received_vectors.pop(port, None)
        self._recompute()
        self._advertise()  # Also teaches a new neighbor when our best routes persist.

    def handle_remove_link(self, port):
        self.adjacencies.pop(port, None)
        self.received_vectors.pop(port, None)
        self._recompute()
        self._advertise()

    def handle_time(self, time_ms):
        if time_ms - self.last_time >= self.heartbeat_time:
            self.last_time = time_ms
            self._advertise()

    def __repr__(self):
        rows = [f"DVrouter(addr={self.addr}, infinity={self.INFINITY})"]
        for destination, (cost, port) in sorted(self.forwarding.items()):
            rows.append(f"  {destination}: cost={cost}, port={port}")
        return "\n".join(rows)
