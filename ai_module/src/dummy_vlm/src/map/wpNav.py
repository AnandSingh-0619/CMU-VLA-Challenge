import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
import math
import time

class WaypointController:
    def __init__(self, threshold = 0.65):
        self.current_pose = Pose2D()
        self.threshold = threshold 
        self.success_flag = False
        
        rospy.Subscriber("/state_estimation", Odometry, self.odometry_callback)
        self.waypoint_pub = rospy.Publisher('/way_point_with_heading', Pose2D, queue_size=1000)

    def odometry_callback(self, msg):
        self.current_pose.x = msg.pose.pose.position.x
        self.current_pose.y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        _, _, yaw = self.quaternion_to_euler(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        self.current_pose.theta = yaw

    def quaternion_to_euler(self, x, y, z, w):
        # Convert quaternion to Euler angles (yaw, pitch, roll)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        
        return roll_x, pitch_y, yaw_z

    def publish_waypoints(self, waypoints):
        rate = rospy.Rate(1)  # Control loop rate
        
        for wp in waypoints:
            print("Next Waypoint: ", wp)

            waypoint_msg = Pose2D()
            waypoint_msg.x = wp[0]
            waypoint_msg.y = wp[1]
            waypoint_msg.theta = wp[2] if len(wp) > 2 else 0.0

            # Publish the waypoint
            start_time = time.time()  # Record the start time
            timeout = 30  # Timeout duration in seconds

            while not rospy.is_shutdown():
                current_time = time.time()
                elapsed_time = current_time - start_time  # Calculate elapsed time
                
                distance = self.get_distance_to_waypoint(wp)
                self.waypoint_pub.publish(waypoint_msg)
                print(distance)

                # Check if waypoint is reached or if timeout has occurred
                if distance < self.threshold:
                    rospy.loginfo(f"Waypoint {wp} reached.")
                    break
                elif elapsed_time > timeout:
                    rospy.logwarn(f"Timeout reached. Skipping waypoint {wp}.")
                    break  # Skip to the next waypoint

                rate.sleep()

        # Set the success flag after all waypoints are reached
        self.success_flag = True
        rospy.loginfo("All waypoints reached.")
    def get_distance_to_waypoint(self, wp):
        # Calculate the distance from the current position to the waypoint
        distance = math.sqrt((wp[0] - self.current_pose.x) ** 2 + (wp[1] - self.current_pose.y) ** 2)
        return distance

    def run(self, waypoints):
        self.success_flag = False  # Reset the success flag
        self.publish_waypoints(waypoints)
        return self.success_flag


