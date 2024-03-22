from packet import Packet


class Quelle:
    def __init__(self, name, volume, iat, next_hop, destination):
        self.name = name
        self.volume = volume
        self.iat = iat
        self.next_hop = next_hop
        self.destination = destination
        self.packet_count = 0

    def new_packet(self):
        self.packet_count += 1
        packet = Packet(self.name, self.destination.name, self.packet_count, self.volume, self.destination)
        self.next_hop.put(packet)
