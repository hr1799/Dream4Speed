#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan, Image
from ackermann_msgs.msg import AckermannDriveStamped
from eval_dreamer import DreamerEval
from gymnasium import Env, spaces
import numpy as np

import argparse
import os
import pathlib
import sys
import dreamer.tools as tools
import ruamel.yaml

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

class DreamerNode:
    def __init__(self):
        rospy.init_node('dreamer_node', anonymous=True)

        parser = argparse.ArgumentParser()
        parser.add_argument("--configs", nargs="+")
        parser.add_argument("--expt_name", type=str, required=True)
        args, remaining = parser.parse_known_args()
        yaml = ruamel.yaml.YAML(typ="safe", pure=True)
        configs = yaml.load(
            (pathlib.Path(sys.argv[0]).parent / "dreamer/configs.yaml").read_text()
        )

        name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
        
        defaults = {}
        for name in name_list:
            recursive_update(defaults, configs[name])
        parser = argparse.ArgumentParser()
        
        for key, value in sorted(defaults.items(), key=lambda x: x[0]):
            arg_type = tools.args_type(value)
            parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
        final_config = parser.parse_args(remaining)
        
        # update the above vars in final_config
        final_config.logdir = 'train_logs/' + args.expt_name

        self.lidar_image_size = 128
        self.lidar_image_resolution = 0.1 # meters per pixel
        self.lidar_max_range = 10
        self.lidar_angle_min_deg = -135
        self.lidar_angle_increment_deg = 0.25
        self.MIN_VEL = 0.5
        self.MAX_VEL = 1.2
        self.expt_name = args.expt_name

        self.ego_speed = 0.1
        self.prev_speed = 0.1
        self.steer_avg=0

        observation_space = spaces.Dict()
        observation_space["state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        observation_space["image"] = spaces.Box(low=0, high=255, shape=(self.lidar_image_size, self.lidar_image_size, 3), dtype=np.uint8)
        action_space = spaces.Box(low=-1, high=1, shape=(2,))
            
        self.agent = DreamerEval(final_config, observation_space=observation_space, action_space=action_space)
        self.agent.reset()

        # Subscribe to the /scan topic
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # Publish image
        self.image_pub = rospy.Publisher('/image', Image, queue_size=10)

        # Publish to the /drive topic
        self.drive_pub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10)

    def scan_callback(self, data):
        
        lidar_obs = np.array(data.ranges)

        lidar_image = np.zeros((self.lidar_image_size, self.lidar_image_size, 3), dtype=np.uint8)
        for i in range(len(lidar_obs)):
            if lidar_obs[i] > self.lidar_max_range * 0.95:
                continue
            if lidar_obs[i] < 0:
                continue
            
            angle = self.lidar_angle_min_deg + i * self.lidar_angle_increment_deg
            angle_rad = np.deg2rad(angle)
            x = int(lidar_obs[i] * np.cos(angle_rad) / self.lidar_image_resolution)
            y = int(lidar_obs[i] * np.sin(angle_rad) / self.lidar_image_resolution + self.lidar_image_size / 2)
            if x >= 0 and x < self.lidar_image_size and y >= 0 and y < self.lidar_image_size:
                lidar_image[y, x, :] = 255
                   
        obs_dict = {
            "state": np.zeros(1, dtype=np.float64),
            "image": lidar_image,
            "is_first": False,
            "is_last": False,
            "is_terminal": False
        }

        action = self.agent.get_action(obs_dict)
        steer_now = action[1] #*1 + self.steer_avg*0.0
        self.steer_avg = steer_now
        
        # For this example, let's just publish a constant drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        if self.expt_name == "austria_vel0_01":
            drive_msg.drive.speed = 0.8
        else:
            drive_msg.drive.speed = ((action[0] + 1) / 2.0) *(self.MAX_VEL - self.MIN_VEL) + self.MIN_VEL

        drive_msg.drive.steering_angle = steer_now*0.4

        # Publish the drive message
        self.drive_pub.publish(drive_msg)
        print("Published drive message Velocity: ", drive_msg.drive.speed, "Steering Angle: ", drive_msg.drive.steering_angle)
        
        # Publish the image
        image_msg = Image()
        image_msg.header.stamp = rospy.Time.now()
        image_msg.height = self.lidar_image_size
        image_msg.width = self.lidar_image_size
        image_msg.encoding = "rgb8"
        lidar_image = np.flipud(lidar_image) # just for viz
        image_msg.data = lidar_image.tobytes()
        self.image_pub.publish(image_msg)

def main():
    try:
        dreamer_node = DreamerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
