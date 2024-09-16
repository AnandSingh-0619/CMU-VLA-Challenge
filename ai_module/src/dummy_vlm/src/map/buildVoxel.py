import numpy as np
import rospy
import math
import pickle
import open3d as o3d
import torch
from transformers import AutoProcessor, OwlViTModel
import torch.nn.functional as F
from voxel_map.voxel import VoxelizedPointcloud
from visualization_msgs.msg import MarkerArray
import threading

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = OwlViTModel.from_pretrained('google/owlvit-base-patch32').to(device)
preprocessor = AutoProcessor.from_pretrained('google/owlvit-base-patch32')

class SemanticVoxelMap:
    def __init__(self):
        self.voxel_map = VoxelizedPointcloud()
        self.markers = {}
        self.lock = threading.Lock()
        self.voxel_map_path = 'voxel_map.pkl'

    def marker_callback(self, msg):
        with self.lock:
            for marker in msg.markers:
                marker_id = marker.id
                if marker_id not in self.markers:  # Only process new markers
                    self.markers[marker_id] = {
                        'position': [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],
                        'scale': [marker.scale.x, marker.scale.y, marker.scale.z],
                        'orientation': [marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w],
                        'heading': marker.pose.orientation.w,  # Assuming heading is the w component of orientation
                        'class_name': marker.ns  # Assuming 'ns' is the namespace containing class names
                    }

                    # Process and add the new marker to the voxel map
                    self.process_marker(marker_id)

    def process_marker(self, marker_id):
        marker_data = self.markers[marker_id]
        point = np.array([marker_data['position']])
        scale = np.array([marker_data['scale']])
        orientation = np.array([marker_data['orientation']])
        heading = np.array([marker_data['heading']])
        class_name = np.array([marker_data['class_name']])

        # Convert to tensors
        point_tensor = torch.from_numpy(point.astype(np.float32))
        scale_tensor = torch.from_numpy(scale.astype(np.float32))
        orientation_tensor = torch.from_numpy(orientation.astype(np.float32))
        heading_tensor = torch.from_numpy(heading.astype(np.float32))

        clip_embeddings = self.compute_clip_embeddings(class_name)
        clip_embeddings_tensor = torch.from_numpy(clip_embeddings).to(device)

        # Add new points to the existing voxel map
        weights = torch.ones_like(point_tensor[:, 0])  # Uniform weight of 1 for the new point
        self.voxel_map.add(points=point_tensor, features=clip_embeddings_tensor, rgb=None, weights=None, scale=scale_tensor)

        # Optionally save the updated voxel map periodically
        # self.save_voxel_map(self.voxel_map)

    def compute_clip_embeddings(self, class_names):
        if isinstance(class_names, (np.ndarray, np.generic)):
            class_names = class_names.tolist()
        elif isinstance(class_names, str):
            class_names = [class_names]
        elif not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
            raise TypeError("Input text should be a string, a list of strings or a nested list of strings")

        with torch.no_grad():
            inputs = preprocessor(text=class_names, return_tensors="pt")
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            all_clip_tokens = clip_model.get_text_features(**inputs)
            all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
        
        return all_clip_tokens.cpu().numpy()

    def save_voxel_map(self, voxel_map):
        with open(self.voxel_map_path, 'wb') as file:
            pickle.dump(voxel_map, file)
        print(f"Voxel map saved to {self.voxel_map_path}")

    def visualize_voxel_map(self, voxel_map):
        # voxel_map = voxel_map
        pcd = o3d.geometry.PointCloud()
        points, features, _, _, _ = voxel_map.get_pointcloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = np.random.rand(len(points), 3)  # Random colors
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

    def load_voxel_map(self, file_path):
        with open(file_path, 'rb') as file:
            voxel_map = pickle.load(file)
        return voxel_map