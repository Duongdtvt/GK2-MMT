"""Link-state flooding and shortest paths over advertised outgoing edges."""

from heapq import heappop, heappush
from itertools import count
from json import dumps, loads

from packet import Packet
from router import Router


class LSrouter(Router):
    def __init__(self, addr, heartbeat_time):
        Router.__init__(self, addr)
        self.heartbeat_time = heartbeat_time
        self.last_time = 0
        self.adjacencies = {}  # port -> (neighbor address, outgoing cost)
        self.local_sequence = 0
        self.link_states = {addr: (0, {})}  # origin -> (sequence, outgoing edges)
        self.forwarding = {}  # destination -> (total cost, outgoing port)

    def _send_state(self, port, origin):
        sequence, edges = self.link_states[origin]
        content = dumps({"origin": origin, "sequence": sequence, "edges": edges})
        neighbor = self.adjacencies[port][0]
        self.send(port, Packet(Packet.ROUTING, self.addr, neighbor, content))

    def _publish_local(self):
        edges = {}
        for neighbor, cost in self.adjacencies.values():
            edges[neighbor] = min(cost, edges.get(neighbor, float("inf")))
        self.local_sequence += 1
        self.link_states[self.addr] = (self.local_sequence, edges)
        for port in self.adjacencies:
            self._send_state(port, self.addr)

    def _shortest_paths(self):
        """Dijkstra, carrying the first local port along each candidate path.

        Advertisements describe directed edges: reverse costs need not match.
        Clients have no advertisement and therefore act as terminal vertices.
        """
        best = {}
        pending = []
        ticket = count()
        for port, (neighbor, cost) in self.adjacencies.items():
            # Equal-cost paths prefer the same first neighbor, then the same port.
            rank = (cost, neighbor, port)
            heappush(pending, (rank, next(ticket), neighbor))

        while pending:
            rank, _, vertex = heappop(pending)
            if vertex == self.addr or vertex in best:
                continue
            best[vertex] = rank
            _, edges = self.link_states.get(vertex, (0, {}))
            distance, first_neighbor, first_port = rank
            for neighbor, cost in edges.items():
                if neighbor != self.addr and neighbor not in best:
                    next_rank = (distance + cost, first_neighbor, first_port)
                    heappush(pending, (next_rank, next(ticket), neighbor))

        self.forwarding = {
            destination: (rank[0], rank[2]) for destination, rank in best.items()
        }

    def handle_packet(self, port, packet):
        if packet.is_traceroute:
            entry = self.forwarding.get(packet.dst_addr)
            if entry is not None:
                self.send(entry[1], packet)
            return
        if not packet.is_routing or port not in self.adjacencies:
            return

        update = loads(packet.content)
        origin = update["origin"]
        if origin == self.addr:
            return  # Only this router can originate its own adjacency list.
        previous_sequence, previous_edges = self.link_states.get(origin, (-1, {}))
        if update["sequence"] <= previous_sequence:
            return
        self.link_states[origin] = (update["sequence"], update["edges"])
        if update["edges"] != previous_edges:
            self._shortest_paths()
        # Even unchanged newer states must propagate beyond this router.
        for outgoing_port in self.adjacencies:
            if outgoing_port != port:
                self._send_state(outgoing_port, origin)

    def handle_new_link(self, port, endpoint, cost):
        self.adjacencies[port] = (endpoint, cost)
        self._publish_local()
        self._shortest_paths()
        # Synchronize previously learned origins across a newly joined link.
        # Each packet still contains exactly one origin's adjacency list.
        for origin in self.link_states:
            if origin != self.addr:
                self._send_state(port, origin)

    def handle_remove_link(self, port):
        self.adjacencies.pop(port, None)
        self._publish_local()
        self._shortest_paths()

    def handle_time(self, time_ms):
        if time_ms - self.last_time >= self.heartbeat_time:
            self.last_time = time_ms
            self._publish_local()

    def __repr__(self):
        rows = [f"LSrouter(addr={self.addr}, sequence={self.local_sequence})"]
        for destination, (cost, port) in sorted(self.forwarding.items()):
            rows.append(f"  {destination}: cost={cost}, port={port}")
        return "\n".join(rows)
