from quelle import Quelle
from senke import Senke
from queue import PriorityQueue


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    senke = Senke("SuperSenke")
    quelle = Quelle("A1", 1500, 2, senke, senke)
    queue = PriorityQueue()

    packet = quelle.new_packet()

    tupel1 = (2, 1, 1, quelle.new_packet, None)

    queue.put(tupel1)





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
