from queue import PriorityQueue

from Labor_1.event import Event
from Labor_1.event_q import EventQ

'''
Die Simulation ist ein globale Datenstruktur, die Attribute
 (Simulationszeit, Ereignisliste, Ereigniszähler)
 und Methoden (run, add_event) besitzt.
Die Simulation ist quasi ein Objekt, das es nur einmal gibt
'''
'''
class TabeleRow:
    def __init__(self, t, n, fun, args):
        self.prio = PriorityQueue()  # Ereignisliste
        self.n = n  # Ereigniszähler
        self.fun = fun  # new Packet
        self.args = args  # Ziel des Pakets

 '''

# Ereigniszähler:
global n
n = 0

# Ereignisliste:
global EventQ
EventQ = EventQ()

# Simulationszeit:
global t
t = 0


def increase_counter(self, counter):
    global n
    n = counter
    n += 1


def add_event(event):
    global EventQ
    EventQ.put(event)


def run(T):
    while not EventQ.empty() and t < T:
        # GET the event from queue
        event = EventQ.get()

        event.process_event()


# def print_new(self):
#     while not self.queue.empty():
#         row = self.queue.get()
#         print(f"{row[0]}, {row[1]}, {row[2]}, {row[3].name}, {row[4]}")
