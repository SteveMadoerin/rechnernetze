from queue import PriorityQueue


class TabeleRow:
    def __init__(self, t, n, fun, args):
        self.t = t
        self.prio = PriorityQueue()
        self.n = n
        self.fun = fun
        self.args = args


class Simulation:
    def __init__(self):
        self.queue = PriorityQueue()

    def create_simulation_definition(self, row):
        self.queue.put(row)

    def print(self):
        while not self.queue.empty():
            row = self.queue.get()
            print(f"{row[0]}, {row[1]}, {row[2]}, {row[3].name}, {row[4]}")
