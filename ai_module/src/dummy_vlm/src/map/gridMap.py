import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2

class GridMapHandler:
    def __init__(self, origin=(0, 0)):
        self.grid_map = None
        self.origin = origin
        self.map_ready = False

    def create_2d_map(self, points, resolution=0.25):
        points_2d = np.array([[point[0], -1*point[1]] for point in points])

        min_x = np.min(points_2d[:, 0])
        max_x = np.max(points_2d[:, 0])
        min_y = np.min(points_2d[:, 1])
        max_y = np.max(points_2d[:, 1])
        # print(min_x, min_y, max_x, max_y)
        buffer = 0  # Buffer of 0.5 meters
        min_x -= buffer
        max_x += buffer
        min_y -= buffer
        max_y += buffer

        height = int(np.ceil((max_x - min_x) / resolution))
        width = int(np.ceil((max_y - min_y) / resolution))
        # print(f"Grid Map Dimensions: Width={width}, Height={height}")
        grid_map = np.zeros((height, width))
    
        for point in points_2d:
            x_idx = int((point[0] - min_x) / resolution)
            y_idx = int((point[1] - min_y) / resolution)
            
            if 0 <= x_idx < height and 0 <= y_idx < width:
                # print("Setting the grid to 1")
                grid_map[ x_idx, y_idx] = 1

        return grid_map, (min_x, min_y)

    def pointcloud_callback(self, msg):
        points_list = []
        for point in pc2.read_points(msg, skip_nans=True):
            points_list.append([point[0], point[1], point[2]])

        if points_list:
            self.grid_map, self.origin = self.create_2d_map(points_list)
            self.map_ready = True

    def get_grid_map(self):
        return self.grid_map, self.origin

    def is_map_ready(self):
        return self.map_ready

