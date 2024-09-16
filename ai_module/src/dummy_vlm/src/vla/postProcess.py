# postProcess.py
import re 
from vla.publisher import publish_object_marker
from vla.publisher import publish_waypoints
import heapq
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque

def find_global_and_scale(localized_points, target_grid_coord):
        for obj in localized_points:
            obj_name = obj['object']
            points = obj['point']
            scales = obj['scale']
            grid_coords = obj['grid_point']  # This should already be stored in your localized_points structure
            
            for idx, (grid_coord, point, scale) in enumerate(zip(grid_coords, points, scales)):
                if grid_coord == target_grid_coord:
                    return {
                        'name': obj_name,
                        'point': point,  # Global coordinates
                        'scale': scale  # Scale data
                    }
        
        return None 
def plot_grid_and_path(path, grid_map, origin):
    filename='grid_map_path.png'


    plt.figure(figsize=(10, 10))
    plt.imshow(grid_map, cmap='Greys', origin='lower', interpolation='none')
    plt.colorbar(label='Occupancy')
    resolution = 0.25
    # Convert path coordinates to match the grid map
    path_y = [p[0]  for p in path]
    path_x = [p[1]  for p in path]
    
    plt.plot(path_x, path_y, 'r-o', markersize=5, label='Path')
    plt.legend()
    plt.title('Grid Map with Smoothed Path')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.grid(True)
    plt.show()
    plt.savefig(filename)
    plt.close()

    
def is_traversable(grid_map, point):
        x, y = int(point[0]), int(point[1])
        return 0 <= x < (grid_map.shape[0]) and 0 <= y < (grid_map.shape[1]) and grid_map[x, y] != 0


def line_of_sight(grid, start, end):
    """
    Checks if there's an unobstructed line of sight between two points in the grid.
    Uses Bresenham's line algorithm or another similar method.
    """
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while (x0, y0) != (x1, y1):
        if grid[x0][y0] == 0:
            return False  # Line intersects an obstacle
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return True

def theta_star(grid, start, goal):
    """
    Theta* pathfinding algorithm to find the shortest path from start to goal.
    
    :param grid: 2D grid map where 1 is traversable and 0 is an obstacle.
    :param start: Tuple (x, y) representing the starting grid coordinate.
    :param goal: Tuple (x, y) representing the goal grid coordinate.
    :return: List of waypoints (grid coordinates) representing the shortest, smoothest path from start to goal.
    """
    def heuristic(point):
        return math.sqrt((point[0] - goal[0])**2 + (point[1] - goal[1])**2)

    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: heuristic(start)}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for neighbor in get_all_neighbors(grid, current):
            # Check if there's line-of-sight between the current node and the grandparent (parent's parent)
            if came_from[current] and line_of_sight(grid, came_from[current], neighbor):
                # Direct connection to grandparent for smoother path
                tentative_g_score = g_score[came_from[current]] + math.sqrt(
                    (neighbor[0] - came_from[current][0])**2 + (neighbor[1] - came_from[current][1])**2
                )
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = came_from[current]
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
            else:
                # Regular A* logic if no direct line-of-sight
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return []

def get_all_neighbors(grid, current):
    """
    Gets traversable neighbors for a given point.
    Can include diagonals for more flexibility.
    """
    neighbors = []
    x, y = current

    # 8 possible directions (up, down, left, right, and diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy

        # Check bounds and traversability
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 1:
            neighbors.append((nx, ny))

    return neighbors

def bresenham_line(x1, y1, x2, y2):
    # Bresenham's Line Algorithm to get the points between two coordinates
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return points
    
def Goto(x, y):
    #return the coordinate as waypoint to be added the list
    # print(f"Executing Goto({x},{y})")
    return (x, y)

def Between(coord1, coord2):
    #return the coordinate as waypoint to be added the list
    # print(f"Executing Between({coord1}, {coord2})")
    midpoint = (int((coord1[0] + coord2[0]) // 2), int((coord1[1] + coord2[1]) // 2))
    return midpoint

def AvoidBetween(coord1, coord2, grid_map):
    #return the updated grid map marking the line between the two points as non traversable
    # print(f"Executing AvoidBetween({coord1}, {coord2})")
    updated_grid_map = grid_map.copy()
    line_points = bresenham_line(coord1[0], coord1[1], coord2[0], coord2[1])
    # Mark the cells along the line (this is just a placeholder, implement your logic)
    for point in line_points:
        x, y = point
        # print(f"({x},{y}) marked as obstacle")
        updated_grid_map[x][y] = 0  # Mark the point as an obstacle

        # Get all neighbors and mark them as obstacles
        neighbors = get_all_neighbors(updated_grid_map, point)
        for nx, ny in neighbors:
            # print(f"Neighbor ({nx},{ny}) marked as obstacle")
            updated_grid_map[nx][ny] = 0  # Mark the neighbor as an obstacle
    return updated_grid_map

def execute_plan(plan, grid_map, origin, current_location):

    # Parse plan into subgoals
    subgoals = {}
    subgoals['0'] = []
    for i in plan.split('\n'):
        i = i.strip()
        if len(i) < 1:
            continue
        if "#" in i:
            sg = i.split("#")[1]
            sg = sg.strip()
            subgoals[sg] = []
        else:
            subgoals['0'].append(i)

    # Begin execution
    executable_steps = 0
    total_steps = 0
    last_assert = None
    wp_list=[]
    # print(subgoals)
    updated_grid = grid_map
    # Execute subgoal '0'
    for action in subgoals['0']:
        # print(action)
        step = 1
        if step > 10:
            break
        try:
            # Process commands and call appropriate functions
            if "Goto" in action:
                # Extract coordinates from the command
                coords = action[action.index("(")+1:action.index(")")].split(',')
                x, y = int(coords[0].strip()), int(coords[1].strip())
                
                point = Goto(x, y)
                if not is_traversable(updated_grid, point):
                    point = find_closest_traversable(point, updated_grid, origin)
                
                wp_list.append(point)
            
            elif "Between" in action:
                # Extract coordinates from the command
                coords = action[action.index("(")+1:action.rindex(")")].split("),(")
                coord1 = tuple(map(int, coords[0].replace('(', '').split(',')))
                coord2 = tuple(map(int, coords[1].replace(')', '').split(',')))
                point = Between(coord1, coord2)
                # print("Midpoint: ", point)
                #check if the point is in traversable area or not. If not find point closest that is in traversable area
                if not is_traversable(updated_grid, point):
                    point = find_closest_traversable(point, updated_grid, origin)
                
                wp_list.append(point)

            elif "Avoidbetween" in action:
                # Extract coordinates from the command
                coords = action[action.index("(")+1:action.rindex(")")].split("),(")
                coord1 = tuple(map(int, coords[0].replace('(', '').split(',')))
                coord2 = tuple(map(int, coords[1].replace(')', '').split(',')))
                updated_grid = AvoidBetween(coord1, coord2, updated_grid)
                
            elif "StopAt" in action:
                # Extract coordinates from the command
                coords = action[action.index("(")+1:action.index(")")].split(',')
                x, y = int(coords[0].strip()), int(coords[1].strip())
                
                point = Goto(x, y)
                if not is_traversable(updated_grid, point):
                    point = find_closest_traversable(point, updated_grid, origin)
                
                wp_list.append(point)
                
        except IndexError:
            print(f"[ERROR] Invalid coordinate format in action: {action}")
        except ValueError:
            print(f"[ERROR] Non-integer values found in action: {action}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {str(e)}")


        step += 1

    waypoints = [current_location] + wp_list
    print(f"Waypoint list: {waypoints}")

    
    full_grid_path = generate_full_path(updated_grid, waypoints, origin)
    # print("Grid_path: ", full_grid_path)
    real_world_waypoints = grid_to_global(full_grid_path, origin)
    
    # print(real_world_waypoints)
    complete_flag =publish_waypoints(real_world_waypoints)
    if complete_flag:
        return "Instruction following completed"
    else:
        return "Instruction incomplete"

def find_closest_traversable(point, grid_map, origin, max_radius=float('inf')):
    # If the point is non-traversable, find the closest traversable point
    x, y = point

    # print(f"Finding closest traversable point to: {point}")
    # Placeholder logic: return the original point for now, implement A* or BFS for real logic
    if is_traversable(grid_map, point):
        return point
    queue = deque([(x, y)])
    visited = set([(x, y)])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # Right, Left, Down, Up, Diagonals

    radius = 1
    while radius<max_radius:
        for _ in range(len(queue)):
            cx, cy = queue.popleft()
            if is_traversable(grid_map,(cx, cy)):
                return (cx, cy)

        for dx, dy in directions:
            for _ in range(radius):
                nx, ny = cx + dx * _, cy + dy * _
                if 0 <= nx < len(grid_map) and 0 <= ny < len(grid_map[0]) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        radius += 1
    # If no traversable point is found within the max radius, return None or handle accordingly
    return None
    
def generate_full_path(grid_map, waypoints, origin, threshold=1):
    """
    Generates a smooth path between waypoints using A* algorithm.

    :param grid_map: 2D grid map where 1 is traversable and 0 is an obstacle.
    :param waypoints: List of waypoints (grid coordinates) that the robot needs to navigate.
    :return: List of all grid coordinates that the robot will follow.
    """
    full_path = []

    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        goal = waypoints[i + 1]

        # Generate A* path between start and goal
        path = theta_star(grid_map, start, goal)
        full_path.extend(path)
    # print(full_path)
    # plot_grid_and_path([i for i in full_path], grid_map, origin)

    return full_path
    
def grid_to_global(waypoints, origin, grid_resolution=0.25):
    """
    Converts grid coordinates to global coordinates.

    :param waypoints: List of tuples representing grid coordinates (x, y).
    :param origin: Tuple representing the global origin (origin_x, origin_y).
    :param grid_resolution: The resolution of the grid in meters per cell. Defaults to 0.25.
    :return: List of tuples representing global coordinates (global_x, global_y).
    """
    global_waypoints = []
    for (x, y) in waypoints:
        global_x = x * grid_resolution + origin[0]
        global_y = -1 * (y * grid_resolution + origin[1])
        global_waypoints.append((global_x, global_y))
    
    return global_waypoints
    
def process_numerical_response(response, localized_points):
    """
    Processes the response for a numerical query.

    Args:
        response (str): The raw response from the VLM.
        localized_points (list): List of localized objects and their grid coordinates.

    Returns:
        dict: Processed numerical response.
    """
    lines = response.splitlines()
    
    # Loop through lines to find the one starting with 'Answer:'
    for line in lines:
        if line.startswith("Answer:"):
            # Extract and return the part after 'Answer:'
            answer = line.split("Answer:")[-1].strip()
            return answer

    return "No answer found"

def process_object_reference_response(response, localized_points):
    """
    Processes the response for an object reference query.

    Args:
        response (str): The raw response from the VLM.
        localized_points (list): List of localized objects and their grid coordinates.

    Returns:
        dict: Processed object reference response.
    """
    # Extract the coordinates from the response
    match =re.search(r'Answer:\s*\(\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\)', response)

    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        target_grid_coord = [x, y]
        # print(target_grid_coord)
        marker = find_global_and_scale(localized_points, target_grid_coord)
        publish_object_marker(marker)
        name = marker.get('name', 'Unknown Object')  # Default name if not found
        point = marker.get('point', [0, 0, 0])  
        answer = f"Object '{name}' is located at coordinates: [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]"
        return answer
        
    else:
        #Print some error
        print(f"Invalid response: {response}")

def process_instruction_following_response(response, localized_points, grid_map, origin, current_location):
    """
    Processes the response for an instruction following query.

    Args:
        response (str): The raw response from the VLM.
        localized_points (list): List of localized objects and their grid coordinates.
        find_global_and_scale_func (function): A function to find global coordinates and scale based on grid coordinates.

    Returns:
        dict: Processed instruction following response.
    """
    if "Answer:" in response:
        answer_start = response.index("Answer:") + len("Answer:")
        answer = response[answer_start:].strip()
        return execute_plan(answer, grid_map, origin, current_location)
    else:
        print("Invalid response")


def apply_postprocess(query_type, response, localized_points, grid_map=None, origin=None, current_location=None):
    """
    Applies the appropriate post-processing based on the query type.

    Args:
        query_type (str): The type of query ('numerical', 'object_reference', 'instruction_following').
        response (str): The raw response from the VLM.
        localized_points (list): List of localized objects and their grid coordinates.
        find_global_and_scale_func (function, optional): A function to find global coordinates and scale. Required for object reference and instruction following queries.

    Returns:
        dict: The processed response.
    """
    if query_type == 'numerical':
        return process_numerical_response(response, localized_points)
    
    elif query_type == 'object_reference':
        return process_object_reference_response(response, localized_points)
    
    elif query_type == 'instruction_following':
        return process_instruction_following_response(response, localized_points, grid_map, origin, current_location)
    
