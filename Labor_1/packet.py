class Packet:
    def __init__(self, source_name, sink_name, number, volume, destination):
        self.name = f"{source_name}-{sink_name}-{number}"
        self.volume = volume
        self.destination = destination

    def print(self):
        print(f"Paket: {self.name}, Volume: {self.volume}, Destination: {self.destination}")

