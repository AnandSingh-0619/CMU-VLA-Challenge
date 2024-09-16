import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import rospy
from scipy.sparse import csr_matrix
import heapq
import math
from scipy.ndimage import binary_dilation
from scipy.ndimage import distance_transform_edt
import time
from sklearn.neighbors import KDTree
from map.wpNav import WaypointController


class Planner:
    def __init__(self, grid_handler, resolution=0.25):
        self.grid_handler = grid_handler
        self.pose = WaypointController()
        self.grid_map = None
        self.origin = (0, 0)
        self.resolution = resolution
        self.min_gap = 0.6
        self.nav_complete = False
        self.safety_buffer = 0.15 
        self.threshold_distance=0.2

    def initialize_grid_map(self):
        self.grid_map, self.origin = self.grid_handler.get_grid_map()
        self.grid_map = csr_matrix(self.grid_map)
        # self.add_safety_buffer()

    def add_safety_buffer(self):
        """Expand non-traversable areas by a buffer zone based on safety_buffer distance."""
        buffer_cells = int(self.safety_buffer / self.resolution)
        dense_grid = self.grid_map.toarray()
        non_traversable_mask = dense_grid == 0  

        # Apply dilation to create a buffer zone
        dilated_mask = binary_dilation(non_traversable_mask, structure=np.ones((buffer_cells, buffer_cells)))
        
        dense_grid[dilated_mask] = 0
        self.grid_map = csr_matrix(dense_grid)

    
    def is_traversable(self, point):
        x, y = int(point[0]), int(point[1])
        return 0 <= x < (self.grid_map.shape[0]) and 0 <= y < (self.grid_map.shape[1]) and self.grid_map[x, y] != 0

    def get_points_to_visit(self):
        height, width = self.grid_map.shape
        min_gap_cells = int(self.min_gap / self.resolution)
        
        # Convert the sparse grid_map to a dense matrix
        dense_grid_map = self.grid_map.toarray() if isinstance(self.grid_map, csr_matrix) else self.grid_map

        # Create the non-traversable mask
        non_traversable_mask = dense_grid_map == 0
        
        # Compute the distance transform on the dense grid map
        distance_from_non_traversable = distance_transform_edt(~non_traversable_mask) * self.resolution
        
        points = []
        for i in range(0, height, min_gap_cells):
            for j in range(0, width, min_gap_cells):
                point = (i, j)
                if self.is_traversable(point):
                    # Check if the point is farther than the threshold distance from any non-traversable area
                    if distance_from_non_traversable[i, j] >= self.threshold_distance:
                        points.append(point)
        
        return points

    def compute_shortest_path_distance(self, start, goal):
        def heuristic(point):
            return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start)}

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return []

    def get_neighbors(self, point):
        x, y = point
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        return [(nx, ny) for nx, ny in neighbors if self.is_traversable((nx, ny))]
    
    def find_nearby_points(self, points, threshold):
        # Convert points to a NumPy array for KDTree
        point_array = np.array(points)
        kdtree = KDTree(point_array)

        nearby_pairs = []
        for i, point in enumerate(points):
            indices = kdtree.query_radius([point], r=threshold)[0]
            for index in indices:
                if index != i:
                    nearby_pairs.append((i, index))

        return nearby_pairs
    

    def find_optimal_path(self, points, threshold):
        
        # Find nearby points within the threshold distance
        nearby_pairs = self.find_nearby_points(points, threshold)
        
        G = nx.Graph()
        for i, j in nearby_pairs:
            start = points[i]
            goal = points[j]
            path = self.compute_shortest_path_distance(start, goal)
            if path:
                G.add_edge(i, j, weight=len(path))
        
        # Solve the TSP using the precomputed distance matrix
        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
    
        return tsp_path

    def calculate_global_coordinates(self, grid_point):
        x_global = self.origin[0] + grid_point[0] * self.resolution
        y_global = self.origin[1] + grid_point[1] * self.resolution
        return (x_global, -y_global)

    def waypoints_from_path(self, path, points, min_distance=1.0):
        waypoints = []
        for i in range(len(path) - 1):
            start_point = points[path[i]]
            goal_point = points[path[i + 1]]
            short_path = self.compute_shortest_path_distance(start_point, goal_point)

            if short_path:
                for grid_point in short_path:
                    waypoints.append(self.calculate_global_coordinates(grid_point))
            else:
                waypoints.append(self.calculate_global_coordinates(goal_point))
        waypoints.append((0,0))

        return self.remove_intermediate_waypoints(waypoints)

    def remove_intermediate_waypoints(self, waypoints):
        if len(waypoints) < 3:
            return waypoints

        filtered_waypoints = [waypoints[0]]  # Always keep the first waypoint

        for i in range(1, len(waypoints) - 1):
            prev_point = waypoints[i - 1]
            curr_point = waypoints[i]
            next_point = waypoints[i + 1]

            # Calculate direction vectors
            vec1 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
            vec2 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])

            # Check if the vectors are collinear (i.e., in the same direction)
            if vec1[0] * vec2[1] != vec1[1] * vec2[0]:
                filtered_waypoints.append(curr_point)

        filtered_waypoints.append(waypoints[-1])  # Always keep the last waypoint

        return filtered_waypoints

    
    def plot_grid_and_path(self, path):
        filename='grid_map_path.png'
        if self.grid_map is None:
            rospy.logerr("Grid map is not initialized.")
            return

        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid_map.toarray(), cmap='Greys', origin='lower', interpolation='none')
        plt.colorbar(label='Occupancy')

        # Convert path coordinates to match the grid map
        path_y = [(p[0] - self.origin[0]) / self.resolution for p in path]
        path_x = [-(p[1] + self.origin[1]) / self.resolution for p in path]
        
        plt.plot(path_x, path_y, 'r-o', markersize=5, label='Path')
        plt.legend()
        plt.title('Grid Map with Smoothed Path')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.grid(True)
        plt.show()
        plt.savefig(filename)
        plt.close()

    def plan_full_coverage_path(self):
        # Initialize the grid map
        t_start=time.time()
        self.initialize_grid_map()
        
        # Define points to visit
        points = self.get_points_to_visit()

        optimal_path_indices = self.find_optimal_path(points,8.0)
        grid_x = int(np.floor((self.pose.current_pose.x - self.origin[0]) / 0.25))
        grid_y = int(np.floor((-1* self.pose.current_pose.y - self.origin[1])/ 0.25))  
        current_grid_point = (grid_x, grid_y)

        # Find the closest point to the robot's current position in the points list
        closest_point_index = None
        min_distance = float('inf')
        for i, point in enumerate(points):
            distance = math.sqrt((point[0] - current_grid_point[0]) ** 2 + (point[1] - current_grid_point[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_point_index = i

        # Insert the closest point index at the start of the optimal path
        optimal_path_indices.insert(0, closest_point_index)
        # Convert path to waypoints
        waypoints = self.waypoints_from_path(optimal_path_indices, points)
        t_end = time.time() - t_start
        print("Got Waypoints in ", t_end)

        # print(waypoints)
        # self.plot_grid_and_path([(wp[0], wp[1]) for wp in waypoints])
        return waypoints

