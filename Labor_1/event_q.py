from queue import PriorityQueue

from Labor_1.event import Event


class EventQ(PriorityQueue):

    class EventQ(PriorityQueue):
        def __init__(self):
            super().__init__()  # Initialize the parent class

        # Store event - put(event):
        def put(self, event: Event):
            # The Event class has implemented ordering, so we can directly put the event
            super().put(event)

        # Return the next event - event=get():
        # The next event is sorted by the criteria of time, priority, event number.
        def get(self):
            # Directly return the event object
            return super().get()



