from quelle import Quelle, globaleSimulation
from simulation import Simulation
from senke import Senke

if __name__ == '__main__':
    def start(simulationsdauer):
        senke = Senke("SuperSenke")
        quelle = Quelle("Quelle", 1500, 2, senke, senke)
        quelle.new_packet(simulationsdauer)

    start(20)
    print("-----ErgebnisListe--------")
    globaleSimulation.print()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/









