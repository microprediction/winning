from winning.lattice import state_prices_from_events
import random
import time


if __name__=='__main__':
    st = time.time()
    n = 125
    scores = [random.choice([-5,1,4,2]) for _ in range(n) ]
    typical_event = [0,0, 0, 0.3, 0.45, 0.1, 0.05,  0.05,  0.05]
    assert sum(typical_event)==1
    events = list()
    for k in range(n):
        k_events = list()
        for _ in range(18):
            k_events.append([ p for p in typical_event])
        events.append( k_events )

    prices = state_prices_from_events(scores=scores, events=events, L=250, unit=1.0)
    print({'elapsed':time.time()-st})
    print(prices)






