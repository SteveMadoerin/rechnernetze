from Labor_1.quelle import Quelle
from Labor_1.senke import Senke
from event import Event
from simulation import Simulation

if __name__ == '__main__':

    def create_packet(source):
        source.new_packet()

    # Define Senke and Quelle object
    sink1 = Senke("S1")
    source1 = Quelle("Q1", 10, 5, sink1, sink1)
    simulation = Simulation(duration=20000)
    event_creation_order = 0

    # Create Events at system start
    for _ in range(10):  # 10 Events as example
        event_creation_order += 1
        simulation.schedule_event(
            Event(time=0, priority=1, creation_order=event_creation_order, function=create_packet, args=(source1,)))

    # start Simulation with interval = 2
    simulation.run(2)



