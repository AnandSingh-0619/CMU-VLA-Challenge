from voxel_map.voxel_map_localizer import VoxelMapLocalizer
import numpy as np
from vla.queryVLM import QueryVLM
from map.gridMap import GridMapHandler
from vla.queryGenerator import generate_query
from vla.postProcess import apply_postprocess
from map.wpNav import WaypointController
import time

class QuestionHandler:
    def __init__(self, model_name):
        self.grid_map_handler = GridMapHandler()
        self.vlm = QueryVLM(model_name)        
        self.pose = WaypointController()
        self.localizer = None
        self.localized_points = []
        self.semantic_map = None
        self.semantic_map_channels = {}
        self.grid_map =None
        self.grid_map_boundary =None
        self.localized_points = None
        self.origin =None
        
    def extract_objects_of_interest(self, instruction):
        prompt = (f"Extract the objects of interest from the following instruction and return them as a list of strings. "
                  f"The objects of interest are any items or entities mentioned in the instruction. "
                  f"Ensure that each item is in its singular form. Do not change the way it is spelled in the instruction. Response should be strings separated by commas.\n\n"
                  f"Instruction: \"{instruction}\"")
        response = self.vlm.query(prompt)
        return response.split(',') if response else []

    def localize_objects(self, objects_of_interest):
        localized_points = []
        for obj in objects_of_interest:
            obj = obj.strip()
            if obj:
                target_point, target_scale = self.localizer.localize_AonB(obj, "", k_A=10, k_B=50, threshold=0.85, data_type='xyz')
                grid_points = self.convert_global_to_grid(target_point.cpu())
                localized_points.append({'object': obj,
                                         'point': target_point.tolist(),
                                         'grid_point': [list(p) for p in grid_points],
                                         'scale': target_scale.tolist()})
        return localized_points

    def convert_global_to_grid(self, global_points):
        # Define your grid resolution and origin
        self.grid_map, origin = self.grid_map_handler.get_grid_map()
        grid_size = (self.grid_map.shape[0], self.grid_map.shape[1])
        cell_size = 0.25 # resolution, adjust as needed
        grid_points=[]
        for coord in global_points:

                grid_x = int(np.floor((coord[0] - origin[0]) / cell_size))
                grid_y = int(np.floor((-1*coord[1] - origin[1])/ cell_size))
                grid_points.append((grid_x, grid_y))

        return grid_points

    def format_localized_points(self, scale_threshold=0.6):
        formatted_str = ""

        for obj in self.localized_points:
            obj_name = obj['object']
            points = obj['point']
            scales = obj['scale']
            grid_coords = obj.get('grid_point', [])  # Retrieve grid coordinates if available

            formatted_str += f"Object: {obj_name} detected at {len(points)} locations:\n"

            for idx, (point, scale) in enumerate(zip(points, scales), start=1):
                x, y, z = float(point[0]), float(point[1]), float(point[2])

                # Get grid coordinates, default to 'N/A' if not available
                if idx - 1 < len(grid_coords):
                    gx, gy = grid_coords[idx - 1]
                    gx_str, gy_str = f"{gx}", f"{gy}"
                else:
                    gx_str, gy_str = 'N/A', 'N/A'

                # Determine scale information
                if scale[0] > scale_threshold and scale[1] > scale_threshold:
                    grid_scale_x = int(np.ceil(scale[0] / 0.25))
                    grid_scale_y = int(np.ceil(scale[1] / 0.25))
                    x_min = max(gx - grid_scale_x // 2, 0)
                    y_min = max(gy - grid_scale_y // 2, 0)
                    x_max = min(gx + grid_scale_x // 2, self.grid_map.shape[0])
                    y_max = min(gy + grid_scale_y // 2, self.grid_map.shape[1])
                    scale_info = f"is enclosed within a bounding box defined by the following corners: Bottom-left: ({x_min}, {y_min}), Top-left: ({x_min}, {y_max}), Top-right: ({x_max}, {y_max}), Bottom-right: ({x_max}, {y_min})."

                else:
                    scale_info = ""

                formatted_str += (
                    f"{obj_name} {idx}:- Centered at ({gx_str},{gy_str}), {scale_info}\n"
                )

            formatted_str += "\n"

        return formatted_str


    def handle_numerical(self, question, info):
        print(f"Handling numerical question: {question}")
        
        query = generate_query(info, self.grid_map_boundary, question,self.grid_map, query_type="numerical")
        
        # print(query)
        response = self.vlm.query(query)
        print(f"API Response: {response}")
        return apply_postprocess('numerical', response, self.localized_points)
        
        
    def handle_object_reference(self, question, info):
        print(f"Handling object_reference question: {question}")
        query = generate_query(info, self.grid_map_boundary, question,self.grid_map, query_type="object_reference")
        # print(query)
        response = self.vlm.query(query)
        print(f"API Response: {response}")
        return apply_postprocess('object_reference', response, self.localized_points)


    def handle_instruction_following(self, question, info):
        print(f"Handling instruction_following question: {question}")
        query = generate_query(info, self.grid_map_boundary, question, self.grid_map, query_type="instruction_following")
        # print(query)
        response = self.vlm.query(query)
        print(f"API Response: {response}") 

        grid_x = int(np.floor((self.pose.current_pose.x - self.origin[0]) / 0.25))
        grid_y = int(np.floor((-1* self.pose.current_pose.y - self.origin[1])/ 0.25))    
        return apply_postprocess('instruction_following', response,  self.localized_points, self.grid_map, self.origin, (grid_x, grid_y))

    def process_question(self, question, voxel_map):
        self.localizer = VoxelMapLocalizer(voxel_map, device='cpu')
        self.grid_map, self.origin = self.grid_map_handler.get_grid_map()
        self.grid_map_boundary = {
            'x_min': 0,
            'y_min': 0,
            'x_max': self.grid_map.shape[0],
            'y_max': self.grid_map.shape[1]
        }
        
        
        prompt = (f"""You are tasked with classifying questions into one of three types based on their content. Below are descriptions of the three types of questions you need to classify:

        Numerical Questions:
            These questions ask about the quantity of objects that fit certain attributes or spatial relations.
            The response is expected to be an integer number.
            Examples:
                How many blue chairs are between the table and the wall?
                How many black trash cans are near the window?

        Object Reference Questions:
            These questions ask the system to find a specific object located in the scene based on spatial relations and/or attributes.
            The response is expected to be a bounding box around the object, with the center point used as a waypoint to navigate the robot system.
            Examples:
                Find the potted plant on the kitchen island that is closest to the fridge.
                Find the orange chair between the table and sink that is closest to the window.

        Instruction-Following Questions:
            These questions instruct the system to take a certain path or trajectory using objects as references.
            The response is expected to be a sequence of waypoints.
            Examples:
                Take the path near the window to the fridge.
                Avoid the path between the two tables and go near the blue trash can near the window.

        valid_responses = {'numerical', 'object reference', 'instruction following'}
        Your task: Classify the following question into one of the three types listed above. Provide your classification in **only one** of the following formats:: 
        - numerical 
        - object reference 
        - instruction following

        Question: {question}  
        """)
        response = self.vlm.query(prompt).replace('*', '').strip().lower()
        print(response)
        valid_responses = {'numerical', 'object reference', 'instruction following'}
        if response in valid_responses:
            
            objects_of_interest = self.extract_objects_of_interest(question)
            self.localized_points = self.localize_objects(objects_of_interest)
            info = self.format_localized_points()
            if response == 'numerical':
                return self.handle_numerical(question, info)
            elif response == 'object reference':
                return self.handle_object_reference(question, info)
            elif response == 'instruction following':
                return self.handle_instruction_following(question, info)
                
        else:
            print(f"Invalid response: {response}")

       

