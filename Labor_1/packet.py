class Packet:
    def __init__(self, source_name, sink_name, number, size, destination):
        self.name = f"{source_name}-{sink_name}-{number}" # String mit Format Source-Sink-Nummer
        self.size = size  # Packetgroesse in Bit
        self.destination = destination # Ziel des Pakets (kann ein Objekt, ein Name oder eine Adresse sein)

    def print(self):
        print(f"Paket: {self.name}, Size: {self.size}, Destination: {self.destination}")

