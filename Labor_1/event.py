import functools


@functools.total_ordering
class Event:
    def __init__(self, time, priority, creation_order, function, args=()):
        self.time = time
        self.priority = priority
        self.creation_order = creation_order
        self.function = function
        self.args = args

    def __lt__(self, other):
        return (self.time, self.priority, self.creation_order) < (other.time, other.priority, other.creation_order)

    def __eq__(self, other):
        return (self.time, self.priority, self.creation_order) == (other.time, other.priority, other.creation_order)

    def process_event(self):
        self.function(*self.args)
