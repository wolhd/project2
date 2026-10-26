from collections import defaultdict

def group_by_time(items):
    d = defaultdict(list)
    for it in items:
        d[it.time].append(it)
    return d

def find_matches(A, B, C, start, similar):
    b_by_t = group_by_time(B)
    c_by_t = group_by_time(C)

    matched = []   # (a, [similar b's], [similar c's])
    for a in A:
        if a.time < start:
            continue
        good_b = [b for b in b_by_t.get(a.time, ()) if similar(a, b)]
        good_c = [c for c in c_by_t.get(a.time, ()) if similar(a, c)]
        if good_b or good_c:
            matched.append((a, good_b, good_c))
    return matched
# usage
# matched = find_matches(A, B, C, start=1_700_000_000, similar=my_similarity_test)
# for a, bs, cs in matched:
#    ...

# estimator const vel
import numpy as np
import pymap3d as pm

def rhumb_propagate(r, v, dt, step=10.0):
    lat, lon, h = pm.ecef2geodetic(*r)                 # degrees
    e, n, u = pm.ecef2enuv(*v, lat, lon)               # constant ENU velocity
    k = max(1, int(np.ceil(abs(dt) / step)))
    d = dt / k
    for _ in range(k):
        lat, lon, h = pm.enu2geodetic(e*d, n*d, u*d, lat, lon, h)
    return np.array(pm.geodetic2ecef(lat, lon, h)), np.array(pm.enu2ecefv(e, n, u, lat, lon))
