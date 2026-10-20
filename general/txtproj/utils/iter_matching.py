
import numpy as np
import bisect
class State:
    def __init__(self,t,v):
        self.time = t
        self.val = v
    def __repr__(self):
        return f"({self.time},{self.val})"

def find_idx(objlist, time):
    getfunc = lambda o: o.time
    idx = bisect.bisect_left(objlist, time, key=getfunc)
    if idx < len(objlist) and objlist[idx].time == time:
        return idx
    return -1

def test():
    ts = np.arange(1.1, 10.1, 1.0)
    dets = [State(x,x) for x in ts]
    fa = [State(x,x) for x in ts[3:]]
    fb = [State(x,x) for x in ts[2:5]]
    print(ts, dets, fa, fb)
    flist = [fa, fb]
    # iterate matching dets
    for i in range(len(flist)):
        ff = flist[i]
        f_idx = 0
        start_time = ff[f_idx].time
        det_idx = find_idx(dets, start_time)
        if det_idx < 0:
            print(f"unexpected cannot find det time {start_time}")
        while f_idx < len(ff):
            f = ff[f_idx]
            d = dets[det_idx]
            # do op
            f_idx += 1
            det_idx += 1
            print(f"check det val {d.val}, f val {f.val}")


test()
