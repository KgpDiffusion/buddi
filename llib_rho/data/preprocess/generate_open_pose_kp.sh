# WARNING -> Estimated time to run this script is 37 mins
# change directory to openpose
cd /home/user/Abhinav/openpose
# Generate open pose detections for Behave
./build/examples/openpose/openpose.bin --image_dir /home/user/Abhinav/behave/train/images --face --hand --write_json /home/user/Abhinav/behave/train/openpose --display 0 --render_pose 0
./build/examples/openpose/openpose.bin --image_dir /home/user/Abhinav/behave/val/images --face --hand --write_json /home/user/Abhinav/behave/val/openpose --display 0 --render_pose 0
./build/examples/openpose/openpose.bin --image_dir /home/user/Abhinav/behave/test/images --face --hand --write_json /home/user/Abhinav/behave/test/openpose --display 0 --render_pose 0

# Generate open pose detections for intercap
./build/examples/openpose/openpose.bin --image_dir /home/user/Abhinav/intercap/train/images --face --hand --write_json /home/user/Abhinav/intercap/train/openpose --display 0 --render_pose 0
./build/examples/openpose/openpose.bin --image_dir /home/user/Abhinav/intercap/val/images --face --hand --write_json /home/user/Abhinav/intercap/val/openpose --display 0 --render_pose 0
./build/examples/openpose/openpose.bin --image_dir /home/user/Abhinav/intercap/test/images --face --hand --write_json /home/user/Abhinav/intercap/test/openpose --display 0 --render_pose 0