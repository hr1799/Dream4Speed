#!/bin/bash

SESSION_NAME="hardware"

if tmux has-session -t "$SESSION_NAME" >/dev/null 2>&1; then
    # If session exists, attach to it
    tmux attach-session -t "$SESSION_NAME"
else
    # Launch tmux session
    tmux new-session -d -s hardware

    # Split window into 2x2 grid
    tmux split-window -v
    tmux select-pane -t 0
    tmux split-window -h
    tmux select-pane -t 2
    tmux split-window -h

    # Run commands in each pane (add sleeps to wait for roscore to start)
    # tmux send-keys -t 1 "sr2; cd /home/lx_hardware/ros2_ws && colcon build && sr2 && ros2 launch lx_bringup_hardware bringup_hardware.launch.py" C-m
    # tmux send-keys -t 4 "sr1; roscore" C-m
    # tmux send-keys -t 5 "bash" C-m
    # sleep 8;
    # tmux send-keys -t 0 "sr1; roslaunch lx_arduino_handler arduino_handler.launch" C-m
    # tmux send-keys -t 2 "sr1; roslaunch husky_launch husky_launch.launch" C-m
    # tmux send-keys -t 3 "sr1; rosparam load /home/lx_hardware/ros2_ws/src/lx_packages/bridge.yaml; sr2; ros2 run ros1_bridge parameter_bridge" C-m

    # Launch roscore, ros1_bridge, and ros2 nodes
    tmux send-keys -t 0 "sr1; roscore" C-m
    tmux send-keys -t 1 "sr1;  sleep 5; rosparam load /home/dreamerv3/ros/bridge.yaml; sr2; source /home/ros2_ws/install/setup.bash; \
                                ros2 run ros1_bridge parameter_bridge" C-m
    # tmux send-keys -t 2 "sr1; cd /home/dreamerv3; python3.10 dreamer_node.py --expt_name austria_vel0_01_smoothsteer_speed" C-m
    tmux send-keys -t 2 "sr1; cd /home/dreamerv3; python3.10 dreamer_node.py --expt_name austria_vel0_01" C-m
    # tmux send-keys -t 3 "sr2; cd /home/dreamerv3/ros; python3 ros2_test.py" C-m

    # Attach to tmux session
    tmux attach-session -t hardware
fi