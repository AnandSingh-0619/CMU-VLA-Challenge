import rospy
from visualization_msgs.msg import Marker    
from geometry_msgs.msg import Pose2D
import time
from map.wpNav import WaypointController

def publish_object_marker(object):
    
    # Create a publisher for the marker
    marker_pub = rospy.Publisher('/selected_object_marker', Marker, queue_size=10)
    
    # Create a Marker message
    marker = Marker()
    marker.header.frame_id = "map"  # Reference frame
    marker.header.stamp = rospy.Time.now()
    marker.ns = object['name']
    marker.id = 0
    marker.type = 1 # Assuming a cube for bounding box
    marker.action = 0

    # Set the position of the marker
    marker.pose.position.x = object['point'][0]
    marker.pose.position.y = object['point'][1]
    marker.pose.position.z = object['point'][2]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Set the scale (bounding box size)
    marker.scale.x = object['scale'][0]
    marker.scale.y = object['scale'][1]
    marker.scale.z = object['scale'][2]

    # Set the color of the marker
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.4  # Alpha (transparency)

    # marker_pub.publish(marker)

    # Publish the marker
    start_time = rospy.Time.now()

    # Publish the marker for 5 seconds
    while not rospy.is_shutdown():
        current_time = rospy.Time.now()
        if (current_time - start_time).to_sec() > 5:
            break
        marker_pub.publish(marker)
        rospy.sleep(0.1)  # Sleep for 0.1 seconds to avoid excessive publishing



def publish_waypoints(waypoints):
    controller = WaypointController(threshold = 0.8)

    navigation_successful = controller.run(waypoints)
       
    return navigation_successful