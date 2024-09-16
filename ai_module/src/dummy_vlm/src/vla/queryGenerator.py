# queryGenerator.py
def generate_query(info, grid_map_boundary, instruction, grid_map, query_type="numerical"):
    if query_type == "numerical":
        query = (
            f"""You are an intelligent assistive robot with access to localized points of objects of interest within an environment.
            Example: How many pillows are on the sofa.
        	Reasoning: The sofa is centered at (27, 36) and covers grid coordinates from (21, 34) to (33, 38). The pillows are centered at (30, 36), (23, 36), (25, 36), and (28, 36), with each pillow covering specific grid coordinates. The grid coordinates of all four pillows overlap with the grid coordinates of the sofa.
            Answer:4

            Environment Context:
            - The environment contains various objects, each with a specific type, location, and size.
            - Environment boundary: 
                X range: {grid_map_boundary['x_min']} to {grid_map_boundary['x_max']} 
                Y range: {grid_map_boundary['y_min']} to {grid_map_boundary['y_max']}
            - You have localized points for these objects.
                   
            Based on the grid map and the localized objects, please analyze the provided information to understand the geometrical and spatial arrangement of the objects within the environment 
            and answer the question.  
            Localized Points of Objects of Interest:
            While mapping, you saw the layout of the house; you saw the following objects with their Global location in environment and as grid coordinates of the map.
            {info}          
            Question: {instruction}            
            Use chain of thought reasoning as in above example to support your answer.
            Response Format:
            Provide the answer as a single integer.
            Please ONLY respond in the format: 
            Reasoning: reason about the given answer and
            Answer: your answer
            """
        )
    elif query_type == "object_reference":
        query = (
            f"""You are an intelligent assistive robot with access to localized points of objects of interest within an environment.                      
            Example: Find the potted plant near the book on the cabinet
        	Reasoning:
        	The question asks for the potted plant near the book on the cabinet. The localized points of objects of interest show that there are 6 potted plants and 6 books in the environment.
        	The cabinet is at grid coordinates (33, 26) and there are six possible locations for the potted plant [(9,51),(9,51),(19,26),(48,27),(36,26),(23,51)] and six possible locations for the book [(26,51),(26,51),(26,51),(26,51),(35,26),(25,51)].
            Therefore, there are 36 possible combinations of potted plant and book near the cabinet.
        	We need to find the combination where the potted plant is geometrically near the book on the cabinet in terms of grid coordinates.
        	Answer: Potted plant at grid coordinates (36, 26)  


            Environment Context:
            - The environment contains various objects, each with a specific type, location, and size.
            - Environment boundary: 
                X range: {grid_map_boundary['x_min']} to {grid_map_boundary['x_max']} 
                Y range: {grid_map_boundary['y_min']} to {grid_map_boundary['y_max']}
            - You have localized points for these objects.
                   
            Based on the grid map and the localized objects, please analyze the provided information to understand the geometrical and spatial arrangement of the objects within the environment 
            and answer the question.            
            Localized Points of Objects of Interest:
            While mapping, you saw the layout of the house; you saw the following objects with their Global location in environment and as grid coordinates of the map.
            {info}
 
            Question: {instruction}  
            Use chain of thought reasoning as in above example

            Your answer should be the UNIQUE coordinates of only the target object from the list of localized points provided.
            Please ONLY respond in the format: 
            Answer: (x,y)
            """
        )

    elif query_type == "instruction_following":
       
        query = (f"""You are an intelligent assistive robot with access to localized points of objects of interest within an environment.


            Environment Context:
            - The environment contains various objects, each with a specific type, location, and size.
            - Environment boundary: 
                X range: {grid_map_boundary['x_min']} to {grid_map_boundary['x_max']} 
                Y range: {grid_map_boundary['y_min']} to {grid_map_boundary['y_max']}
            - You have localized points for these objects.

                   
            Based on the grid map and the localized objects, please analyze the provided information to understand the geometrical and spatial arrangement of the objects within the environment 
            and answer the question.            
            Localized Points of Objects of Interest:
            While mapping, you saw the layout of the house; you saw the following objects with their Global location in environment and as grid coordinates of the map.
            {info}
 
            Question: {instruction}  
            Use chain of thought reasoning, as demonstrated in the example above.
            Break down the task into a series of task and then classify them in below possible placeholders
            action_list[Goto(x,y), Between((x1,y1),(x2,y2)), Avoidbetween((x1,y1),(x2,y2), StopAt(x,y)]
            Please ONLY respond in the format: RESPONSE FORMAT: 
            -Reasoning: Explain the reasoning behind the answer, including how the target objects and path were determined.
            -Answer: output should be a list of these placeholders in  order of instruction followiing
            This sould follow the order of naviagtion as in instruction
                    
            
            Example Output Format:
            - Reasoning: [explain your reasoning here]
            - Answer: 
                Goto(2,2)
                Between((24,15),(32,15))
                Goto(21,28)
                StopAt(2,47)
               
            """)
            
    else:
        query = "Unsupported query type."

    return query
