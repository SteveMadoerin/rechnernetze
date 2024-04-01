import functools


@functools.total_ordering
class Event:
    def __init__(self, t, prio, n, func, args):
        self.time = t
        self.priority = prio
        self.number = n
        self.function = func
        self.args = args

    def __lt__(self, other):
        return (self.time, self.priority, self.number) < (other.time, other.priority, other.number)

    def __eq__(self, other):
        return (self.time, self.priority, self.number) == (other.time, other.priority, other.number)

    # abarbeitungsfunktion
    def process_event(self):
        self.function(*self.args)
