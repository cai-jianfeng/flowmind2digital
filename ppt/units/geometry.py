import math

def calc_dis(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def cmin(x, y, px, py, dis, direction, d):
    cur = calc_dis(x, y, px, py)
    if cur < dis:
        dis = cur
        direction = d
    return dis, direction

def calc(x, y, shape):
    """
    'circle': 0,
    'diamonds': 1,
    'long_oval': 2,
    'hexagon': 3,
    'parallelogram': 4,
    'rectangle': 5,
    'trapezoid': 6,
    'triangle': 7,
    """
    cls = shape[4]
    dis = 1e18
    direction = 0

    xmin, ymin, xmax, ymax = shape[0:4]
    px, py = xmin, (ymin + ymax) / 2
    dis, direction = cmin(x, y, px, py, dis, direction, 1)
    px, py = (xmin + xmax) / 2, ymin
    dis, direction = cmin(x, y, px, py, dis, direction, 0)
    px, py = xmax, (ymin + ymax) / 2
    dis, direction = cmin(x, y, px, py, dis, direction, 3)
    px, py = (xmin + xmax) / 2, ymax
    dis, direction = cmin(x, y, px, py, dis, direction, 2)

    return dis, direction
