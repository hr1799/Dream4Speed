#!/usr/bin/env python3

import rospy
from ackermann_msgs.msg import AckermannDriveStamped

def main():
    # Initialize the ROS node with a name
    rospy.init_node('chatter_publisher', anonymous=True)
    
    # Create a publisher to the 'chatter' topic with String message type
    pub = rospy.Publisher('drive', AckermannDriveStamped, queue_size=10)
    
    # Set a publishing rate of 1 Hz
    rate = rospy.Rate(1)  # 1 Hz
    
    while not rospy.is_shutdown():
        # The message to be published
        message = AckermannDriveStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = 'base_link'
        message.drive.steering_angle = 0.5
        message.drive.speed = 1.0
        
        # Publish the message to the 'chatter' topic
        rospy.loginfo(f"Publishing: {message}")
        pub.publish(message)
        
        # Sleep to maintain the publishing rate
        rate.sleep()

if __name__ == '__main__':
    main()
