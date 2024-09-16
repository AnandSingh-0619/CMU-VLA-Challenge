import torch
import torch.nn.functional as F
import os
import pickle
import clip
from voxel_map.voxel import VoxelizedPointcloud
import rospy
from visualization_msgs.msg import Marker
from transformers import AutoProcessor, OwlViTModel

class VoxelMapLocalizer():
    def __init__(self, voxel_map,device =None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
 
        self.model_name = 'google/owlvit-base-patch32'
        self.clip_model, self.preprocessor = self.load_pretrained()#"ViT-B/32", device=self.device)
        # file_path = 'voxel_map.pkl'
        # self.voxel_pcd = self.load_voxel_map(file_path)

        self.voxel_pcd = voxel_map


    def load_pretrained(self):
        # As mentioned, we only support owlvit-base-patch32 for now
        model = OwlViTModel.from_pretrained(self.model_name).to(self.device)
        preprocessor = AutoProcessor.from_pretrained(self.model_name)
        return model, preprocessor

    def load_voxel_map(self, file_path):
        with open(file_path, 'rb') as file:
            voxel_map_data = pickle.load(file)

        voxel_pcd = VoxelizedPointcloud()
        voxel_pcd.add(points=voxel_map_data._points,
                    features=voxel_map_data._features,
                    rgb=voxel_map_data._rgb,
                    weights=voxel_map_data._weights,
                    scale=voxel_map_data._scale)
        return voxel_pcd
    

    def calculate_clip_and_st_embeddings_for_queries(self, queries):
        with torch.no_grad():
            # We only support owl-vit for now
            inputs = self.preprocessor(
                text=[queries], return_tensors="pt"
            )
            inputs['input_ids'] = inputs['input_ids'].to(self.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
            all_clip_tokens = self.clip_model.get_text_features(**inputs)
            all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
        return all_clip_tokens
        
    def find_alignment_over_model(self, queries):
        clip_text_tokens = self.calculate_clip_and_st_embeddings_for_queries(queries)
        points, features, _, _,_ = self.voxel_pcd.get_pointcloud()
        features = F.normalize(features, p=2, dim=-1)
        if features is not None:
            features = features.to(self.device)

        point_alignments = clip_text_tokens.detach() @ features.T
    
        # print(point_alignments.shape)
        return point_alignments

    # Currently we only support compute one query each time, in the future we might want to support check many queries

    def localize_AonB(self, A="cushion", B="Couch", k_A = 10, k_B = 50, threshold=0.5 ,data_type = 'r3d'):
        # print("A is ", A)
        # print("B is ", B)
        if B is None or B == '':
            target_pos, target_scale = self.find_alignment_for_A([A], threshold=threshold)
        else:
            points, _, _, _ = self.voxel_pcd.get_pointcloud()
            alignments = self.find_alignment_over_model([A, B]).cpu()
            A_points = points[alignments[0].topk(k = k_A, dim = -1).indices].reshape(-1, 3)
            B_points = points[alignments[1].topk(k = k_B, dim = -1).indices].reshape(-1, 3)
            distances = torch.norm(A_points.unsqueeze(1) - B_points.unsqueeze(0), dim=2)
            target = A_points[torch.argmin(torch.min(distances, dim = 1).values)]
        if data_type == 'r3d':
            target = target[[0, 2, 1]]
            target[1] = -target[1]
        return target_pos, target_scale

    def find_alignment_for_A(self, A, threshold=0.5):
        points, features, _, _, scale = self.voxel_pcd.get_pointcloud()
        alignments = self.find_alignment_over_model(A).cpu()
        mask = alignments.squeeze() > threshold
        return points[mask].reshape(-1, 3), scale[mask].reshape(-1, 3)

def main():
    rospy.init_node('voxel_map_localizer')

    localizer = VoxelMapLocalizer(device='cpu')
    
    # Example localization
    A = "buddha decoration"
    B = ""
    target_point, target_scale = localizer.localize_AonB(A, B, k_A=10, k_B=50, data_type='xyz')

    marker_pub = rospy.Publisher('/selected_object_marker', Marker, queue_size=10)
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "selected_object"
    marker.id = 0
    marker.type = 1
    marker.action = 0
    marker.pose.position.x = target_point[0].item()
    marker.pose.position.y = target_point[1].item()
    marker.pose.position.z = target_point[2].item()
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = target_scale[0].item()
    marker.scale.y = target_scale[1].item()
    marker.scale.z = target_scale[2].item()
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0

    while not rospy.is_shutdown():
        marker_pub.publish(marker)
        rospy.sleep(1.0)

if __name__ == "__main__":
    main()

