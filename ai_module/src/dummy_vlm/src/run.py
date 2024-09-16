#!/usr/bin/env python

# Author: Anand Mohan Singh
# This script creates a state machine that interacts with ROS for voxel mapping and answering spatial queries.
# It also launches a Flask web server for a GUI to interact with the state machine.
# Default link to access the web interface for GUI: http://localhost:16552

import threading
import time
import rospy
from sensor_msgs.msg import PointCloud2
from map.gridMap import GridMapHandler  
from map.planner import Planner
from map.buildVoxel import SemanticVoxelMap
from map.wpNav import WaypointController
from visualization_msgs.msg import MarkerArray
from voxel_map.voxel import VoxelizedPointcloud
import pickle
from enum import Enum
from vla.handleQuestions import QuestionHandler 
import webbrowser
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import signal
import sys
from rosgraph_msgs.msg import Log
from vla.publisher import publish_waypoints
import json
import os

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

class State(Enum):
    INITIALIZE = 1
    MAPPING = 2
    ASK_QUESTION = 3
    ANSWERING = 4

class StateMachine:
    def __init__(self):
        self.state = State.INITIALIZE
        self.voxel_map = None
        self.nav_complete = False
        self.question = ""
        self.ques_handler = QuestionHandler(model_name="mistral")
        self.logs = []
        self.answer = ""
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        try:
            rospy.loginfo(f"Web interface: http://localhost:16552")
            while not rospy.is_shutdown() and self.running:
                # with self.lock:
                    if self.state == State.INITIALIZE:
                        self.initialize()
                    elif self.state == State.MAPPING:
                        self.mapping()
                    elif self.state == State.ASK_QUESTION:
                        rospy.sleep(1)
                    elif self.state == State.ANSWERING:
                        self.answering()
        except rospy.ROSInterruptException:
            self.cleanup()

    def initialize(self):
        try:
            self.log("Initializing ROS environment...")
            rospy.Subscriber('/traversable_area', PointCloud2, self.ques_handler.grid_map_handler.pointcloud_callback)
            while not self.ques_handler.grid_map_handler.is_map_ready() and self.running:
                rospy.sleep(1)
            self.state = State.MAPPING
        except Exception as e:
            self.log(f"Error initializing ROS environment: {e}")
            self.state = State.INITIALIZE

    def mapping(self):
        try:
            self.log("Waiting for the agent to navigate and map the environment...")
            res = "no"
            if res.lower() == "yes":
                self.load_stored_map()
            else:
                self.create_voxel_map()
            self.state = State.ASK_QUESTION
            self.log("Please type in the Question in the box")
        except Exception as e:
            self.log(f"Error during mapping: {e}")
            self.state = State.INITIALIZE

    def load_stored_map(self):
        try:
            self.log("Loading the stored map and waypoints...")
            with open('voxel_map.pkl', 'rb') as file:
                voxel_map_data = pickle.load(file)
            self.voxel_map = VoxelizedPointcloud()
            self.voxel_map.add(
                points=voxel_map_data._points,
                features=voxel_map_data._features,
                rgb=voxel_map_data._rgb,
                weights=voxel_map_data._weights,
                scale=voxel_map_data._scale
            )
            self.log("Voxel map loaded from file.")
        except (FileNotFoundError, pickle.PickleError) as e:
            self.log(f"Error loading stored map: {e}")
            self.create_voxel_map()

    def create_voxel_map(self):
        try:
            self.log("Starting voxel map creation...")
            planner = Planner(self.ques_handler.grid_map_handler)
            waypoint_list = planner.plan_full_coverage_path()
            voxel_map_handler = SemanticVoxelMap()
            rospy.Subscriber('/object_markers', MarkerArray, voxel_map_handler.marker_callback)
            controller = WaypointController()

            navigation_successful = controller.run(waypoint_list)
            if navigation_successful:
                self.voxel_map = voxel_map_handler.voxel_map
                voxel_map_handler.save_voxel_map(self.voxel_map)
                self.log("Voxel map saved.")
            else:
                self.log("Navigation unsuccessful, retrying...")
                self.state = State.MAPPING
        except Exception as e:
            self.log(f"Error creating voxel map: {e}")
            self.state = State.INITIALIZE

    def ask_question(self, question):
        with self.lock:
            self.question = question
            self.state = State.ANSWERING

    def answering(self):
        try:
            self.log(f"Processing question: {self.question}")
            if not self.voxel_map:
                self.log("Voxel map is not set. Cannot process the question.")
                self.state = State.ASK_QUESTION
                return "Voxel map is not set. Cannot process the question."
            
            self.answer = self.ques_handler.process_question(self.question, self.voxel_map)
            self.log(f"Answer: {self.answer}")

            socketio.emit('answer', {'answer':self.answer}, namespace='/')
            self.state = State.ASK_QUESTION
            self.log("Please type in the Question in the box")
            return self.answer
        except Exception as e:
            self.log(f"Error answering question: {e}")
            self.state = State.ASK_QUESTION
            self.log("Please type in the Question in the box")

    def log(self, message):
        # with self.lock:
        self.logs.append(message)
        rospy.loginfo(message)
        socketio.emit('log_update', {'log': message}, namespace='/')

    def get_current_state(self):
        with self.lock:
            return self.state

    def get_last_answer(self):
        with self.lock:
            return self.answer

    def cleanup(self):
        # with self.lock:
        self.log("Cleaning up resources...")
        self.running = False

        rospy.signal_shutdown("Shutting down state machine")
        self.log("Resources freed and ROS shutdown completed.")

   

sm = StateMachine()

rospy.init_node('state_machine_node')
sm.log("ROS node initialized")
def run_state_machine():
    sm.run()

state_machine_thread = threading.Thread(target=run_state_machine)
state_machine_thread.start()

@socketio.on('connect')
def handle_connect():
    for log in sm.logs:
        emit('log_update', {'log': log})
    sm.logs=[]

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('submit_question')
def handle_question(data):
    question = data.get('question', "")
    # socketio.emit('log_update', {'log': sm.logs}, namespace='/')
    # sm.logs = []
    if question:
        sm.ask_question(question)
        while sm.get_current_state() != State.ASK_QUESTION:
            # for log in sm.logs:
            #     emit('log_update', {'log': log})
            # sm.logs = []

            time.sleep(0.01)
        # emit('answer', {'answer': sm.get_last_answer()})
    else:
        emit('answer', {'answer': "No question provided."})

@socketio.on('go_home')
def handle_go_home():
    sm.log("Go to Home button pressed.")
    # Append home waypoint and publish it to the robot
    waypoints = [(0, 0)]
    publish_waypoints(waypoints)
    
    emit('log_update', {'log': "Navigating to home position (0,0)."})
    
@socketio.on('shutdown_system')
def handle_shutdown():
    sm.cleanup()
    os._exit(0)   # Exit the entire program

def signal_handler(sig, frame):
    sm.cleanup()
    state_machine_thread.join()  # Ensure state machine thread is joined before exit
    os._exit(0)

    
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    def start_server():
        socketio.run(app, host='0.0.0.0', port=16552, debug=False)
    
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    # Add a small delay to ensure the server has started
    # time.sleep(2)

    # Open the web browser automatically
    webbrowser.open_new_tab("http://localhost:16552")
    
    # socketio.start_background_task(periodic_log_updates)

