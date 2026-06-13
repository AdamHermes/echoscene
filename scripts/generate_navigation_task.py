import argparse
import heapq
import json
import math
import os
import random


OBSTACLE_CATEGORIES = {
    "armchair",
    "bed",
    "bookshelf",
    "cabinet",
    "chair",
    "desk",
    "lamp",
    "nightstand",
    "shelf",
    "sofa",
    "table",
    "tv_stand",
    "wardrobe",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a 2D navigation task from a structured EchoScene JSON sidecar."
    )
    parser.add_argument("--scene_json", required=True, help="Path to *_echoscene.json")
    parser.add_argument("--out_dir", default=None, help="Output directory. Defaults beside the scene JSON.")
    parser.add_argument("--resolution", type=float, default=0.05, help="Grid cell size in meters.")
    parser.add_argument("--robot_radius", type=float, default=0.20, help="Obstacle inflation radius in meters.")
    parser.add_argument(
        "--room_margin",
        type=float,
        default=0.75,
        help="Extra floor margin in meters around inferred room bounds.",
    )
    parser.add_argument(
        "--bounds_source",
        choices=["auto", "floor", "room_bounds"],
        default="auto",
        help="Use floor object bounds when available, or fall back to room_bounds.",
    )
    parser.add_argument("--preview_scale", type=int, default=4, help="Pixels per occupancy grid cell in the PPM preview.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start_x", type=float, default=None)
    parser.add_argument("--start_z", type=float, default=None)
    parser.add_argument("--goal_x", type=float, default=None)
    parser.add_argument("--goal_z", type=float, default=None)
    parser.add_argument("--min_goal_distance", type=float, default=1.0)
    parser.add_argument("--planner", choices=["rrt", "astar", "cost_astar"], default="rrt")
    parser.add_argument("--num_trials", type=int, default=50, help="Number of sampled navigation trials.")
    parser.add_argument(
        "--trial_mode",
        choices=["random", "farthest", "component_centers"],
        default="random",
        help="How to choose start/goal pairs when explicit coordinates are not provided.",
    )
    parser.add_argument("--rrt_iterations", type=int, default=2000)
    parser.add_argument("--rrt_step_size", type=float, default=0.25, help="RRT extension step in meters.")
    parser.add_argument("--rrt_goal_sample_rate", type=float, default=0.10)
    parser.add_argument(
        "--cost_sigma",
        type=float,
        default=0.35,
        help="Gaussian falloff distance in meters for PhyScene-style object proximity cost.",
    )
    parser.add_argument(
        "--cost_weight",
        type=float,
        default=8.0,
        help="Strength of object proximity cost for --planner cost_astar.",
    )
    parser.add_argument("--show_cost_map", action="store_true", help="Render Gaussian cost shading in the preview.")
    return parser.parse_args()


def load_scene(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def output_paths(scene_json, out_dir, scan_id):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(scene_json)), "navigation_tasks")
    os.makedirs(out_dir, exist_ok=True)
    return (
        os.path.join(out_dir, f"{scan_id}_navigation_task.json"),
        os.path.join(out_dir, f"{scan_id}_occupancy.ppm"),
    )


def floor_bounds_from_scene(scene):
    for obj in scene.get("objects", []):
        if obj.get("category") != "floor":
            continue
        aabb = obj.get("aabb")
        if not aabb:
            continue
        min_x, _, min_z = aabb["min"]
        max_x, _, max_z = aabb["max"]
        if min_x < max_x and min_z < max_z:
            return {
                "min": [float(min_x), 0.0, float(min_z)],
                "max": [float(max_x), 0.0, float(max_z)],
                "source": "floor_object_aabb",
            }
    return None


def get_room_bounds(scene, margin=0.0, bounds_source="auto"):
    floor_bounds = floor_bounds_from_scene(scene)
    if bounds_source == "floor" and floor_bounds is None:
        raise ValueError("Requested --bounds_source floor, but no usable floor object AABB was found.")

    if bounds_source in {"auto", "floor"} and floor_bounds is not None:
        bounds = floor_bounds
    else:
        bounds = scene.get("room_bounds")

    if not bounds:
        raise ValueError("Scene JSON has no room_bounds. Re-export the scene with the structured exporter.")

    min_x, _, min_z = bounds["min"]
    max_x, _, max_z = bounds["max"]
    if min_x >= max_x or min_z >= max_z:
        raise ValueError("Invalid room_bounds in scene JSON.")
    return float(min_x) - margin, float(max_x) + margin, float(min_z) - margin, float(max_z) + margin


def grid_shape(min_x, max_x, min_z, max_z, resolution):
    width = int(math.ceil((max_x - min_x) / resolution)) + 1
    height = int(math.ceil((max_z - min_z) / resolution)) + 1
    return width, height


def world_to_grid(x, z, min_x, min_z, resolution):
    gx = int(round((x - min_x) / resolution))
    gz = int(round((z - min_z) / resolution))
    return gx, gz


def grid_to_world(gx, gz, min_x, min_z, resolution):
    return min_x + gx * resolution, min_z + gz * resolution


def empty_grid(width, height):
    return [[0 for _ in range(width)] for _ in range(height)]


def clamp(value, low, high):
    return max(low, min(high, value))


def mark_rect(grid, min_gx, max_gx, min_gz, max_gz):
    height = len(grid)
    width = len(grid[0])
    min_gx = clamp(min_gx, 0, width - 1)
    max_gx = clamp(max_gx, 0, width - 1)
    min_gz = clamp(min_gz, 0, height - 1)
    max_gz = clamp(max_gz, 0, height - 1)

    for gz in range(min_gz, max_gz + 1):
        for gx in range(min_gx, max_gx + 1):
            grid[gz][gx] = 1


def is_obstacle_object(obj):
    category = obj.get("category")
    if category in {"_scene_", "floor"}:
        return False
    if obj.get("included_in_scene_mesh") is False:
        return False
    return category in OBSTACLE_CATEGORIES or "scene_obstacle" in obj.get("affordances", [])


def rasterize_obstacles(scene, grid, min_x, min_z, resolution, robot_radius):
    obstacles = []
    inflate_cells = int(math.ceil(robot_radius / resolution))
    for obj in scene.get("objects", []):
        if not is_obstacle_object(obj):
            continue
        aabb = obj.get("aabb")
        if not aabb:
            continue

        min_gx, min_gz = world_to_grid(aabb["min"][0], aabb["min"][2], min_x, min_z, resolution)
        max_gx, max_gz = world_to_grid(aabb["max"][0], aabb["max"][2], min_x, min_z, resolution)
        min_gx -= inflate_cells
        max_gx += inflate_cells
        min_gz -= inflate_cells
        max_gz += inflate_cells
        mark_rect(grid, min_gx, max_gx, min_gz, max_gz)
        obstacles.append(
            {
                "id": obj.get("id"),
                "category": obj.get("category"),
                "aabb": aabb,
            }
        )
    return obstacles


def distance_to_aabb_2d(x, z, aabb):
    min_x, min_z = aabb["min"][0], aabb["min"][2]
    max_x, max_z = aabb["max"][0], aabb["max"][2]
    dx = max(min_x - x, 0.0, x - max_x)
    dz = max(min_z - z, 0.0, z - max_z)
    return math.hypot(dx, dz)


def build_cost_map(grid, obstacles, min_x, min_z, resolution, sigma, weight):
    sigma = max(float(sigma), 1e-6)
    weight = max(float(weight), 0.0)
    height = len(grid)
    width = len(grid[0])
    cost_map = [[1.0 for _ in range(width)] for _ in range(height)]

    for gz in range(height):
        for gx in range(width):
            if grid[gz][gx] != 0:
                cost_map[gz][gx] = float("inf")
                continue

            x, z = grid_to_world(gx, gz, min_x, min_z, resolution)
            proximity_cost = 0.0
            for obstacle in obstacles:
                distance = distance_to_aabb_2d(x, z, obstacle["aabb"])
                proximity_cost += math.exp(-(distance * distance) / (2.0 * sigma * sigma))
            cost_map[gz][gx] = 1.0 + weight * proximity_cost
    return cost_map


def free_cells(grid):
    cells = []
    for gz, row in enumerate(grid):
        for gx, value in enumerate(row):
            if value == 0:
                cells.append((gx, gz))
    return cells


def connected_components(grid):
    height = len(grid)
    width = len(grid[0])
    visited = set()
    components = []

    for gz in range(height):
        for gx in range(width):
            if grid[gz][gx] != 0 or (gx, gz) in visited:
                continue
            queue = [(gx, gz)]
            visited.add((gx, gz))
            component = []
            for cell in queue:
                component.append(cell)
                for nx, nz in ((cell[0] + 1, cell[1]), (cell[0] - 1, cell[1]), (cell[0], cell[1] + 1), (cell[0], cell[1] - 1)):
                    if nx < 0 or nz < 0 or nx >= width or nz >= height:
                        continue
                    if grid[nz][nx] != 0 or (nx, nz) in visited:
                        continue
                    visited.add((nx, nz))
                    queue.append((nx, nz))
            components.append(component)

    components.sort(key=len, reverse=True)
    return components


def component_center(component):
    cx = sum(cell[0] for cell in component) / float(len(component))
    cz = sum(cell[1] for cell in component) / float(len(component))
    return min(component, key=lambda cell: (cell[0] - cx) ** 2 + (cell[1] - cz) ** 2)


def farthest_pair(cells):
    if len(cells) < 2:
        raise ValueError("Need at least two cells to choose a farthest pair.")
    first = cells[0]
    a = max(cells, key=lambda cell: heuristic(first, cell))
    b = max(cells, key=lambda cell: heuristic(a, cell))
    return a, b


def nearest_free_cell(grid, target):
    width = len(grid[0])
    height = len(grid)
    start = (clamp(target[0], 0, width - 1), clamp(target[1], 0, height - 1))
    if grid[start[1]][start[0]] == 0:
        return start

    queue = [start]
    visited = {start}
    for gx, gz in queue:
        for nx, nz in ((gx + 1, gz), (gx - 1, gz), (gx, gz + 1), (gx, gz - 1)):
            if nx < 0 or nz < 0 or nx >= width or nz >= height or (nx, nz) in visited:
                continue
            if grid[nz][nx] == 0:
                return nx, nz
            visited.add((nx, nz))
            queue.append((nx, nz))
    return None


def choose_start_goal(grid, min_x, min_z, resolution, rng, min_goal_distance):
    cells = free_cells(grid)
    if len(cells) < 2:
        raise ValueError("No navigable free space after obstacle rasterization.")

    for _ in range(400):
        start = rng.choice(cells)
        goal = rng.choice(cells)
        sx, sz = grid_to_world(start[0], start[1], min_x, min_z, resolution)
        gx, gz = grid_to_world(goal[0], goal[1], min_x, min_z, resolution)
        if math.hypot(gx - sx, gz - sz) >= min_goal_distance:
            return start, goal

    start = cells[0]
    goal = max(cells, key=lambda cell: (cell[0] - start[0]) ** 2 + (cell[1] - start[1]) ** 2)
    return start, goal


def choose_start_goal_by_mode(grid, min_x, min_z, resolution, rng, min_goal_distance, mode):
    if mode == "random":
        return choose_start_goal(grid, min_x, min_z, resolution, rng, min_goal_distance)

    components = connected_components(grid)
    if not components:
        raise ValueError("No navigable free space after obstacle rasterization.")

    if mode == "component_centers" and len(components) >= 2:
        return component_center(components[0]), component_center(components[1])

    return farthest_pair(components[0])


def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar(grid, start, goal):
    width = len(grid[0])
    height = len(grid)
    open_heap = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    neighbors = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (-1, -1, math.sqrt(2.0)),
    ]

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return reconstruct_path(came_from, current)

        for dx, dz, cost in neighbors:
            nxt = (current[0] + dx, current[1] + dz)
            if nxt[0] < 0 or nxt[1] < 0 or nxt[0] >= width or nxt[1] >= height:
                continue
            if grid[nxt[1]][nxt[0]] != 0:
                continue

            tentative = g_score[current] + cost
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                priority = tentative + heuristic(nxt, goal)
                heapq.heappush(open_heap, (priority, nxt))
    return None


def cost_astar(grid, cost_map, start, goal):
    width = len(grid[0])
    height = len(grid)
    open_heap = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    neighbors = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (-1, -1, math.sqrt(2.0)),
    ]

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return reconstruct_path(came_from, current)

        current_cost = cost_map[current[1]][current[0]]
        for dx, dz, step_distance in neighbors:
            nxt = (current[0] + dx, current[1] + dz)
            if nxt[0] < 0 or nxt[1] < 0 or nxt[0] >= width or nxt[1] >= height:
                continue
            if grid[nxt[1]][nxt[0]] != 0:
                continue

            next_cost = cost_map[nxt[1]][nxt[0]]
            traversal_cost = step_distance * 0.5 * (current_cost + next_cost)
            tentative = g_score[current] + traversal_cost
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                priority = tentative + heuristic(nxt, goal)
                heapq.heappush(open_heap, (priority, nxt))
    return None


def line_is_free(grid, start, end):
    x0, z0 = start
    x1, z1 = end
    dx = abs(x1 - x0)
    dz = abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sz = 1 if z0 < z1 else -1
    err = dx - dz

    while True:
        if z0 < 0 or x0 < 0 or z0 >= len(grid) or x0 >= len(grid[0]) or grid[z0][x0] != 0:
            return False
        if x0 == x1 and z0 == z1:
            return True
        err2 = 2 * err
        if err2 > -dz:
            err -= dz
            x0 += sx
        if err2 < dx:
            err += dx
            z0 += sz


def nearest_node(nodes, sample):
    return min(nodes, key=lambda node: heuristic(node, sample))


def steer(from_node, to_node, step_cells):
    dist = heuristic(from_node, to_node)
    if dist <= step_cells:
        return to_node
    ratio = step_cells / dist
    gx = int(round(from_node[0] + (to_node[0] - from_node[0]) * ratio))
    gz = int(round(from_node[1] + (to_node[1] - from_node[1]) * ratio))
    return gx, gz


def trace_tree_path(parent, node):
    path = [node]
    while parent[node] is not None:
        node = parent[node]
        path.append(node)
    path.reverse()
    return path


def join_rrt_paths(parent_a, node_a, parent_b, node_b):
    path_a = trace_tree_path(parent_a, node_a)
    path_b = trace_tree_path(parent_b, node_b)
    path_b.reverse()
    if path_a and path_b and path_a[-1] == path_b[0]:
        path_b = path_b[1:]
    return path_a + path_b


def rrt_connect(grid, start, goal, rng, max_iterations, step_cells, goal_sample_rate):
    if start == goal:
        return [start]

    free = free_cells(grid)
    if not free:
        return None

    tree_a = [start]
    tree_b = [goal]
    parent_a = {start: None}
    parent_b = {goal: None}

    for iteration in range(max_iterations):
        if rng.random() < goal_sample_rate:
            sample = goal if iteration % 2 == 0 else start
        else:
            sample = rng.choice(free)

        nearest_a = nearest_node(tree_a, sample)
        new_a = steer(nearest_a, sample, step_cells)
        if new_a == nearest_a or grid[new_a[1]][new_a[0]] != 0 or not line_is_free(grid, nearest_a, new_a):
            tree_a, tree_b = tree_b, tree_a
            parent_a, parent_b = parent_b, parent_a
            continue

        tree_a.append(new_a)
        parent_a[new_a] = nearest_a

        nearest_b = nearest_node(tree_b, new_a)
        current_b = nearest_b
        while True:
            new_b = steer(current_b, new_a, step_cells)
            if new_b == current_b or grid[new_b[1]][new_b[0]] != 0 or not line_is_free(grid, current_b, new_b):
                break
            tree_b.append(new_b)
            parent_b[new_b] = current_b
            current_b = new_b
            if line_is_free(grid, current_b, new_a):
                if current_b != new_a:
                    tree_b.append(new_a)
                    parent_b[new_a] = current_b
                    current_b = new_a
                path = join_rrt_paths(parent_a, new_a, parent_b, current_b)
                if path[0] == goal:
                    path.reverse()
                return path
            if current_b == new_a:
                path = join_rrt_paths(parent_a, new_a, parent_b, current_b)
                if path[0] == goal:
                    path.reverse()
                return path

        tree_a, tree_b = tree_b, tree_a
        parent_a, parent_b = parent_b, parent_a

    return None


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_length_meters(path_world):
    total = 0.0
    for prev, cur in zip(path_world, path_world[1:]):
        total += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
    return total


def path_to_world(path_grid, min_x, min_z, resolution):
    if path_grid is None:
        return []
    return [
        {
            "x": x,
            "z": z,
        }
        for x, z in (
            grid_to_world(gx, gz, min_x, min_z, resolution)
            for gx, gz in path_grid
        )
    ]


def plan_path(grid, cost_map, start, goal, args, rng):
    if args.planner == "astar":
        return astar(grid, start, goal)
    if args.planner == "cost_astar":
        return cost_astar(grid, cost_map, start, goal)
    step_cells = max(1, int(round(args.rrt_step_size / args.resolution)))
    return rrt_connect(
        grid,
        start,
        goal,
        rng,
        max_iterations=args.rrt_iterations,
        step_cells=step_cells,
        goal_sample_rate=args.rrt_goal_sample_rate,
    )


def write_ppm(path, grid, route=None, start=None, goal=None, scale=1, cost_map=None):
    route = set(route or [])
    width = len(grid[0])
    height = len(grid)
    scale = max(1, int(scale))
    finite_costs = []
    if cost_map is not None:
        for gz in range(height):
            for gx in range(width):
                value = cost_map[gz][gx]
                if math.isfinite(value):
                    finite_costs.append(value)
    max_cost = max(finite_costs) if finite_costs else 1.0

    def cell_color(gx, gz):
        cell = (gx, gz)
        if cell == start:
            return 0, 180, 0
        if cell == goal:
            return 220, 0, 0
        if cell in route:
            return 30, 120, 255
        if grid[gz][gx]:
            return 20, 20, 20
        if cost_map is not None and max_cost > 1.0:
            normalized = (cost_map[gz][gx] - 1.0) / (max_cost - 1.0)
            normalized = max(0.0, min(1.0, normalized))
            red = int(245)
            green = int(245 - 120 * normalized)
            blue = int(245 - 220 * normalized)
            return red, green, blue
        return 245, 245, 245

    with open(path, "w", encoding="ascii") as file:
        file.write(f"P3\n{width * scale} {height * scale}\n255\n")
        for gz in range(height):
            scaled_row = []
            for gx in range(width):
                color = cell_color(gx, gz)
                scaled_row.extend([f"{color[0]} {color[1]} {color[2]}"] * scale)
            row_text = " ".join(scaled_row)
            for _ in range(scale):
                file.write(row_text + "\n")


def cell_component_index(components):
    mapping = {}
    for index, component in enumerate(components):
        for cell in component:
            mapping[cell] = index
    return mapping


def object_reachability(grid, obstacles, min_x, min_z, resolution, components):
    if not components:
        return [], 0.0

    component_ids = cell_component_index(components)
    largest_component_id = 0
    results = []

    for obstacle in obstacles:
        aabb = obstacle["aabb"]
        center_x = (aabb["min"][0] + aabb["max"][0]) * 0.5
        center_z = (aabb["min"][2] + aabb["max"][2]) * 0.5
        nearest = nearest_free_cell(grid, world_to_grid(center_x, center_z, min_x, min_z, resolution))
        reachable = nearest is not None and component_ids.get(nearest) == largest_component_id
        results.append(
            {
                "id": obstacle.get("id"),
                "category": obstacle.get("category"),
                "nearest_free_cell": list(nearest) if nearest is not None else None,
                "reachable_from_largest_component": reachable,
            }
        )

    if not results:
        return results, None
    reachable_count = sum(1 for item in results if item["reachable_from_largest_component"])
    return results, reachable_count / float(len(results))


def build_grid_context(args):
    rng = random.Random(args.seed)
    scene = load_scene(args.scene_json)
    min_x, max_x, min_z, max_z = get_room_bounds(
        scene,
        margin=args.room_margin,
        bounds_source=args.bounds_source,
    )
    width, height = grid_shape(min_x, max_x, min_z, max_z, args.resolution)
    grid = empty_grid(width, height)
    obstacles = rasterize_obstacles(scene, grid, min_x, min_z, args.resolution, args.robot_radius)
    cost_map = build_cost_map(grid, obstacles, min_x, min_z, args.resolution, args.cost_sigma, args.cost_weight)
    return rng, scene, grid, cost_map, obstacles, min_x, min_z


def make_trial(trial_id, grid, cost_map, min_x, min_z, args, rng):
    fixed_start = args.start_x is not None and args.start_z is not None
    fixed_goal = args.goal_x is not None and args.goal_z is not None

    if fixed_start:
        start = nearest_free_cell(grid, world_to_grid(args.start_x, args.start_z, min_x, min_z, args.resolution))
    else:
        start = None

    if fixed_goal:
        goal = nearest_free_cell(grid, world_to_grid(args.goal_x, args.goal_z, min_x, min_z, args.resolution))
    else:
        goal = None

    if start is None or goal is None:
        start, goal = choose_start_goal_by_mode(
            grid,
            min_x,
            min_z,
            args.resolution,
            rng,
            args.min_goal_distance,
            args.trial_mode,
        )

    path_grid = plan_path(grid, cost_map, start, goal, args, rng)
    success = path_grid is not None
    path_world = path_to_world(path_grid, min_x, min_z, args.resolution)
    path_cost = path_traversal_cost(path_grid, cost_map) if success else None
    return {
        "trial_id": trial_id,
        "planner": args.planner,
        "start": {
            "grid": [start[0], start[1]],
            "world": dict(zip(["x", "z"], grid_to_world(start[0], start[1], min_x, min_z, args.resolution))),
        },
        "goal": {
            "grid": [goal[0], goal[1]],
            "world": dict(zip(["x", "z"], grid_to_world(goal[0], goal[1], min_x, min_z, args.resolution))),
        },
        "success": success,
        "path": path_world,
        "path_grid": path_grid or [],
        "metrics": {
            "path_waypoint_count": len(path_world),
            "path_length_m": path_length_meters([(p["x"], p["z"]) for p in path_world]) if success else None,
            "path_cost": path_cost,
            "euclidean_start_goal_m": heuristic(start, goal) * args.resolution,
        },
    }, path_grid, start, goal


def build_task(args):
    rng, scene, grid, cost_map, obstacles, min_x, min_z = build_grid_context(args)
    width = len(grid[0])
    height = len(grid)
    trials = []
    preview_path_grid = None
    preview_start = None
    preview_goal = None

    for trial_id in range(max(1, args.num_trials)):
        trial, path_grid, start, goal = make_trial(trial_id, grid, cost_map, min_x, min_z, args, rng)
        trials.append(trial)
        if preview_path_grid is None or trial["success"]:
            preview_path_grid = path_grid
            preview_start = start
            preview_goal = goal
        if args.start_x is not None and args.start_z is not None and args.goal_x is not None and args.goal_z is not None:
            break

    free_count = len(free_cells(grid))
    total_count = width * height
    components = connected_components(grid)
    largest_component_cells = len(components[0]) if components else 0
    rwalkable = largest_component_cells / float(free_count) if free_count > 0 else 0.0
    object_reachability_items, reachable_object_rate = object_reachability(
        grid,
        obstacles,
        min_x,
        min_z,
        args.resolution,
        components,
    )
    success_count = sum(1 for trial in trials if trial["success"])
    first_trial = trials[0]
    task = {
        "schema_version": "0.1",
        "task_type": "point_navigation",
        "source_scene_json": os.path.abspath(args.scene_json),
        "scan_id": scene.get("scan_id"),
        "planner": {
            "name": args.planner,
            "num_trials": len(trials),
            "trial_mode": args.trial_mode,
            "rrt_iterations": args.rrt_iterations if args.planner == "rrt" else None,
            "rrt_step_size": args.rrt_step_size if args.planner == "rrt" else None,
            "rrt_goal_sample_rate": args.rrt_goal_sample_rate if args.planner == "rrt" else None,
            "cost_sigma": args.cost_sigma if args.planner == "cost_astar" else None,
            "cost_weight": args.cost_weight if args.planner == "cost_astar" else None,
        },
        "grid": {
            "resolution": args.resolution,
            "width": width,
            "height": height,
            "origin": {"x": min_x, "z": min_z},
            "robot_radius": args.robot_radius,
            "room_margin": args.room_margin,
            "bounds_source": args.bounds_source,
            "preview_scale": args.preview_scale,
        },
        "start": first_trial["start"],
        "goal": first_trial["goal"],
        "success": first_trial["success"],
        "path": first_trial["path"],
        "trials": trials,
        "metrics": {
            "success_rate": success_count / float(len(trials)),
            "success_count": success_count,
            "trial_count": len(trials),
            "free_space_ratio": free_count / float(total_count),
            "rwalkable": rwalkable,
            "connected_component_count": len(components),
            "largest_component_cell_count": largest_component_cells,
            "reachable_object_rate": reachable_object_rate,
            "obstacle_cell_count": total_count - free_count,
            "free_cell_count": free_count,
            "mean_success_path_length_m": mean(
                trial["metrics"]["path_length_m"] for trial in trials if trial["success"]
            ),
            "mean_success_path_cost": mean(
                trial["metrics"]["path_cost"] for trial in trials if trial["success"]
            ),
        },
        "obstacles": obstacles,
        "object_reachability": object_reachability_items,
    }
    return task, grid, cost_map, preview_path_grid, preview_start, preview_goal


def path_traversal_cost(path_grid, cost_map):
    if not path_grid:
        return None
    total = 0.0
    for gx, gz in path_grid:
        total += cost_map[gz][gx]
    return total / float(len(path_grid))


def mean(values):
    values = list(values)
    if not values:
        return None
    return sum(values) / float(len(values))


def main():
    args = parse_args()
    task, grid, cost_map, path_grid, start, goal = build_task(args)
    task_path, image_path = output_paths(args.scene_json, args.out_dir, task["scan_id"])

    with open(task_path, "w", encoding="utf-8") as file:
        json.dump(task, file, indent=2)
        file.write("\n")

    preview_cost_map = cost_map if args.show_cost_map else None
    write_ppm(image_path, grid, route=path_grid, start=start, goal=goal, scale=args.preview_scale, cost_map=preview_cost_map)
    print("navigation task exported:", task_path)
    print("occupancy preview exported:", image_path)
    print("success:", task["success"])
    print("planner:", task["planner"]["name"])
    print("success_rate:", "{:.3f}".format(task["metrics"]["success_rate"]))
    print("free_space_ratio:", "{:.3f}".format(task["metrics"]["free_space_ratio"]))
    print("rwalkable:", "{:.3f}".format(task["metrics"]["rwalkable"]))
    if task["metrics"]["reachable_object_rate"] is not None:
        print("reachable_object_rate:", "{:.3f}".format(task["metrics"]["reachable_object_rate"]))
    if task["metrics"]["mean_success_path_length_m"] is not None:
        print("mean_success_path_length_m:", "{:.3f}".format(task["metrics"]["mean_success_path_length_m"]))
    if task["metrics"]["mean_success_path_cost"] is not None:
        print("mean_success_path_cost:", "{:.3f}".format(task["metrics"]["mean_success_path_cost"]))


if __name__ == "__main__":
    main()
