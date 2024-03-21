from paket import Paket
from simulation import Simulation
import time

globaleSimulation = Simulation()


class Quelle:
    def __init__(self, name, volume, iat, next_hop, destination):
        self.name = name
        self.volume = volume
        self.iat = iat
        self.next_hop = next_hop
        self.destination = destination
        self.counter = 0

    def new_packet(self, simulationsdauer):
        newPacket = Paket("A." + str(self.counter), time.sleep(1), self.destination)
        self.next_hop.put(newPacket)
        newPacket.volume
        tupel1 = (self.counter, 1, 1, newPacket, None)
        globaleSimulation.create_simulation_definition(tupel1)
        self.counter += 1
        if self.counter != simulationsdauer:
            self.new_packet(simulationsdauer)
