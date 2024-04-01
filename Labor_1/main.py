from Labor_1.event import Event
from Labor_1.quelle import Quelle
from Labor_1.senke import Senke
import simulation as sim

if __name__ == '__main__':
    # Define sink and source objects
    sink1 = Senke("S1")
    source1 = Quelle("Q1", sink1, 10, 5, sink1)

    # Create an event with the function and arguments, but do not call the function here
    # The function `new_packet` will be called by the simulation when the event is processed
    event = Event(t=0, prio=1, n=1, func=source1.new_packet, args=(sink1,))

    # Add the event to the simulation
    sim.add_event(event)

    # Run the simulation for the specified duration
    sim.run(T=200000)


    # # Define Senke and Quelle object
    # sink1 = Senke("S1")
    # source1 = Quelle("Q1", sink1, 10, 5,  sink1)
    # # source1.new_packet(sink1)
    # event = Event(t=0, prio=1, n=1, func=source1.new_packet(sink1), args=(sink1,))
    # #sim.add_event(event)
    # sim.add_event(0, 1, 1, source1.new_packet, senke=sink1)
    # sim.run(T=200000)


    # Create Events at system start
    # for _ in range(10):  # 10 Events as example
    #     event_creation_order += 1
    #     simulation.schedule_event(
    #         Event(time=0, priority=1, creation_order=event_creation_order, function=create_packet, args=(source1,)))
    #
    # # start Simulation with interval = 2
    # simulation.run(2)





