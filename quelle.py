from paket import Paket
from senke import Senke
from simulation import Simulation

simulation = Simulation()

class Quelle:

    def __init__(self, name, volume, iat, next_hop, destination):
        self.name = name
        self.volume = volume
        self.iat = iat
        self.next_hop = next_hop
        self.destination = destination
        self.counter = 0
        self.max = 5

    def new_packet(self):
        self.counter += 1
        newPacket = Paket(self.name + "." + str(self.counter), self.volume, self.destination)
        self.next_hop.put(newPacket)
        tupel1 = (simulation.t + self.iat, 1, 1, newPacket, None)

        simulation.put(tupel1)
