import queue
import time


class Simulation:
    def __init__(self, duration):
        self.event_queue = queue.PriorityQueue()
        self.current_time = 0
        self.duration = duration

    # PUT the event in queue
    def schedule_event(self, event):
        self.event_queue.put(event)

    def run(self, interval):
        while not self.event_queue.empty() and self.current_time < self.duration:
            # GET the event from queue
            event = self.event_queue.get()
            self.current_time = event.time
            event.process_event()
            time.sleep(interval)
            # Logic for new Events can be implemented here:
