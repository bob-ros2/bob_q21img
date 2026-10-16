#!/bin/bash
set -e

# Source ROS 2 environment
source "/opt/ros/${ROS_DISTRO}/setup.bash"

# Source workspace install if available
if [ -f "/app/ros2_ws/install/setup.bash" ]; then
    source "/app/ros2_ws/install/setup.bash"
fi

exec "$@"
