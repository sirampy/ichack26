import requests
import json
import math
import tkinter as tk
import heapq

DRAW_THRESH = 0.05
SHOW_ROUTE_WHILE_DRAWING = True
DRAWING_TAIL = 5
DEVIATION_PENALTY_GROWTH = 10

#FILE_NAME = "d.json"
#bbox = [ 51.54195639619177, -0.09647369384765626, 51.55551331720391, -0.05321502685546875 ]
FILE_NAME = "e.json"
bbox = [ 51.47747462361918, -0.19964218139648438, 51.51461229219796, -0.1563835144042969 ]


def pedestrian_ways_from_bbox(bbox):
    south, west, north, east = bbox
    overpass_query = f"""
    [out:json][timeout:25];
    (
      way["highway"~"footway|pedestrian|path|steps|track|bridleway|living_street"]
          ({south},{west},{north},{east});
      way["highway"~"residential|tertiary|secondary|primary|unclassified"]["foot"!~"no"]
          ({south},{west},{north},{east});
    );
    out geom;
    """

    overpass_url = "https://overpass-api.de/api/interpreter"

    response = requests.post(
        overpass_url,
        data=overpass_query,
        timeout=60
    )
    response.raise_for_status()

    osm_data = response.json()
    with open(FILE_NAME, "w") as f:
        json.dump(osm_data, f)

#pedestrian_ways_from_bbox(bbox)
#quit()


def normCoord(coord):
    global bbox
    w = bbox[3] - bbox[1]
    h = bbox[2] - bbox[0]
    x = coord["lon"] - bbox[1]
    y = coord["lat"] - bbox[0]
    return [x / w, y / h]

with open(FILE_NAME, "r") as f:
    osm_data = json.load(f)

def rc_enc(c):
    return f"{c[0]},{c[1]}"

def rc_dec(c):
    return [float(i) for i in c.split(",")]

def dist(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)#math.hypot(c1[0], c1[1], c2[0], c2[1])

map_coords = [i["geometry"] for i in osm_data["elements"]]
map_coords = [[rc_enc(normCoord(j)) for j in i] for i in map_coords]

map_coord_conns = {}
for l in map_coords:
    for c_ind in range(len(l)):
        c = l[c_ind]
        # Add prev
        if c_ind != len(l) - 1:
            if c not in map_coord_conns: map_coord_conns[c] = [l[c_ind+1]]
            else:                        map_coord_conns[c].append(l[c_ind+1])
        # Add next
        if c_ind != 0:
            if c not in map_coord_conns: map_coord_conns[c] = [l[c_ind-1]]
            else:                        map_coord_conns[c].append(l[c_ind-1])


# Logic

def connect(start, end, conns, heuristic, drawing):
    heap = [(0, start)]
    distances = {start: 0}
    previous = {}

    while heap:
        current_dist, current_vert = heapq.heappop(heap)

        if current_vert == end: break
        if current_dist > distances.get(current_vert, float("inf")): continue

        for next_vert in conns[current_vert]:
            weighting = dist(rc_dec(current_vert), rc_dec(next_vert))
            new_dist = current_dist + weighting + heuristic(rc_dec(next_vert), drawing)

            if new_dist < distances.get(next_vert, float("inf")):
                distances[next_vert] = new_dist
                previous[next_vert] = current_vert
                heapq.heappush(heap, (new_dist, next_vert))

    # Reconstruct path
    if end not in distances: return

    path = []
    node = end
    while node != start:
        path.append(node)
        if node not in previous: return None
        node = previous[node]
    path.append(start)
    path.reverse()

    return path

def point_to_line_dist(p, a, b):
    x0, y0 = p
    x1, y1 = a
    x2, y2 = b

    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)

    return num / den

def nearest_vert(conns, v):
    min_dist = float("inf")
    for c in conns:
        dist_ = dist(rc_dec(c), v)
        if (dist_ < min_dist):
            min_dist = dist_
            out = c
    return out

def heuristic(v, drawing):
    return min([(1 + point_to_line_dist(v, drawing[i], drawing[i+1])) ** DEVIATION_PENALTY_GROWTH for i in range(len(drawing)-1)])

def get_next_points(cur_start, end, conns, heuristic, drawing):
    route = connect(cur_start, nearest_vert(conns, rc_dec(end)), conns, heuristic, drawing)
    if route is None: return
    out = []
    for v in route[1:]:
        if v == float("inf"): return
        out.append(v)
        if dist(rc_dec(v), drawing[0]) > dist(rc_dec(v), drawing[1]):
            break
    return out


# UI

drawing_vertices = []
canvas_lines = []
fixed_points = []
is_holding = False

def main():
    root = tk.Tk()
    root.title("Test")
    W = 800
    H = 600
    root.geometry(f"{W}x{H}")

    # Create canvas
    canvas = tk.Canvas(
        root,
        width=W,
        height=H,
        bg="white"
    )
    canvas.pack(fill=tk.BOTH, expand=True)

    def drawLine(coords, fill, width):
        if len(coords) <= 1: return
        start = coords[0]
        if type(start) == str: start = rc_dec(start)
        for end in coords[1:]:
            if type(end) == str: end = rc_dec(end)
            canvas.create_line(start[0] * W, start[1] * H, end[0] * W, end[1] * H, fill = fill, width = width)
            start = end

    def drawPoint(p, fill, size):
        if type(p) == str: p = rc_dec(p)
        p[0] *= W
        p[1] *= H
        canvas.create_rectangle(p[0]-size, p[1]-size, p[0]+size, p[1]+size, fill=fill)

    # Draw map
    for l in map_coords:
        drawLine(l, "black", 2)

    def canvas_norm(x, y):
        return [x / W, y / H]

    def press(e):
        global drawing_vertices, is_holding, canvas_lines, fixed_points

        # Clear old sketch data and start new sketch
        drawing_vertices = [canvas_norm(e.x, e.y)]
        is_holding = True

        # Clear old canvas contents
        for l in canvas_lines:
            canvas.delete(l)
        canvas_lines = []

        fixed_points = []

    def move(e):
        global drawing_vertices, is_holding, canvas_lines
        if not is_holding: return
        start = drawing_vertices[-1]
        end = canvas_norm(e.x, e.y)
        if dist(start, end) < DRAW_THRESH: return
        drawing_vertices.append(end)
        canvas_lines.append(canvas.create_line(start[0] * W, start[1] * H, end[0] * W, end[1] * H, fill = "red", width = 10))

        if SHOW_ROUTE_WHILE_DRAWING:
            if len(fixed_points) == 0:
                start = nearest_vert(map_coord_conns, drawing_vertices[0])
            else:
                start = fixed_points[-1]

            route = get_next_points(start, rc_enc(end), map_coord_conns, heuristic, drawing_vertices[-DRAWING_TAIL:])
            if route is None: return

            start = rc_dec(start)
            for v in route[1:]:
                fixed_points.append(v)
                end = rc_dec(v)
                canvas_lines.append(canvas.create_line(start[0] * W, start[1] * H, end[0] * W, end[1] * H, fill = "green", width = 10))
                start = end

    def release(e):
        global drawing_vertices, is_holding, canvas_lines

        is_holding = False

        if SHOW_ROUTE_WHILE_DRAWING:
            if len(fixed_points) == 0:
                start = nearest_vert(map_coord_conns, drawing_vertices[0])
            else:
                start = fixed_points[-1]

            end = nearest_vert(map_coord_conns, drawing_vertices[-1])
            route = connect(start, end, map_coord_conns, heuristic, drawing_vertices[-DRAWING_TAIL:])
            if route is None: return

            start = rc_dec(start)
            for v in route:
                fixed_points.append(v)
                end = rc_dec(v)
                canvas_lines.append(canvas.create_line(start[0] * W, start[1] * H, end[0] * W, end[1] * H, fill = "green", width = 10))
                start = end



    canvas.bind("<Button>", press)
    canvas.bind("<Motion>", move)
    canvas.bind("<ButtonRelease>", release)

    root.mainloop()


if __name__ == "__main__":
    main()
