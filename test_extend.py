import math
def extend_segment(p1, p2, eps=0.1):
    dx = p2[0] - p1[0]
    dz = p2[1] - p1[1]
    dist = math.hypot(dx, dz)
    if dist == 0: return p1, p2
    ux, uz = dx/dist, dz/dist
    return (p1[0] - ux*eps, p1[1] - uz*eps), (p2[0] + ux*eps, p2[1] + uz*eps)
print(extend_segment((0, 0), (1, 0)))
