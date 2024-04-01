from packet import Packet
import simulation as sim


class Quelle:
    def __init__(self, name, next_hop, size, iat, destination):
        self.name = name  # name: String
        self.next_hop = next_hop  # Objekt, an das die Quelle Pakete mit next_hop.put(packet) weitergibt - akt die Senke
        self.size = size  # Groesse der erzeugten Pakete
        self.iat = iat  # zweitliche Abstand der erzeugten Pakete (in Schritt 1a noch nicht benötigt)
        self.destination = destination  # Objekt, Instanz von Sink

        #self.packet_count = 0

    # neues Paket erstellen und an next_hop schicken
    def new_packet(self, senke):
        # global packet count n from modul simulation
        sim.n += 1

        # Packet params: (self, source_name, sink_name, number, size, destination):
        packet = Packet(self.name, self.destination.name, sim.n, self.size, self.destination)
        print(f"Quelle {self.name}: Paket {packet.name} abgeschickt")
        senke.put(packet)
