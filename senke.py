from paket import Paket

class Senke:


    def __init__(self, name):
        self.name = name

    def put(self, p: Paket):
        print(f"Paket {p.name} ist an Senke {self.name} angekommen.")
