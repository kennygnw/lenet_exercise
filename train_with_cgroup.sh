#!/bin/bash

# CONFIG
CGROUP_NAME="lenet_group"
TRAIN_SCRIPT="lenet_train.py"
LOG_FILE="memlog_$(date +%H%M).csv"

# Create cgroup
sudo mkdir -p /sys/fs/cgroup/$CGROUP_NAME
echo +memory | sudo tee /sys/fs/cgroup/cgroup.subtree_control > /dev/null

# Clear previous log
rm -f $LOG_FILE

# Start memory logging in background
(
  while true; do
    ts=$(date +%s)
    mem=$(cat /sys/fs/cgroup/$CGROUP_NAME/memory.current)
    echo "$ts,$((mem/1024/1024))" >> $LOG_FILE
    sleep 1
  done
) &
LOGGER_PID=$!

# Start training script and move it into the cgroup
python3 $TRAIN_SCRIPT &
TRAIN_PID=$!
echo $TRAIN_PID | sudo tee /sys/fs/cgroup/$CGROUP_NAME/cgroup.procs > /dev/null

# Wait for training to finish
wait $TRAIN_PID

# Stop the logger
kill $LOGGER_PID

echo "Training completed. Memory log saved to $LOG_FILE"
