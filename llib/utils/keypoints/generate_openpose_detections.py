# From Python
# It requires OpenCV installed for Python
import sys

sys.path.append("/home/ubuntu/openpose/build/python/openpose")
import pyopenpose as op
import cv2
import os
from sys import platform
import argparse
import time
import glob
import os.path as osp
import subprocess
from tqdm import tqdm
import numpy as np
import json

def move_rename_kp(args, IMGoutdir, json_fn, IMGname):
    os.makedirs(IMGoutdir, exist_ok=True)
    subprocess.call(["mv", json_fn, osp.join(IMGoutdir, IMGname + ".json")])


def initialize_dict():
    """Initialize dictionary with keys for all body parts, face parts and hands.
        
        Returns:
            tmp (dict): dictionary with keys for all body parts, face parts and hands.
    """

    tmp = {}
    tmp["version"] = 1.3
    tmp["people"] = []
    person_dict = {}

    KEYS = [
        "person_id",
        "pose_keypoints_2d",
        "face_keypoints_2d",
        "hand_left_keypoints_2d",
        "hand_right_keypoints_2d",
        "pose_keypoints_3d",
        "face_keypoints_3d",
        "hand_left_keypoints_3d",
        "hand_right_keypoints_3d",
    ]
    
    for key in KEYS:
        if key =="person_id":
            person_dict[key] = [-1]
        else:
            person_dict[key] = []

    tmp["people"].append(person_dict)

    return tmp

def process_keypoints(datum: op.Datum):
    """Extract keypoints from openpose datum object.\
        Select the human with the largest area in the image.
        
        Args:
            datum (op.Datum): openpose datum object
        
        Returns:
            pose_kp (np.array): 2D pose keypoints
            face_kp (np.array): 2D face keypoints
            hand_l_kp (np.array): 2D left hand keypoints
            hand_r_kp (np.array): 2D right hand keypoints"""
    
    SELECTED_HUMAN_IDX=0 #select idx for largest human
    EPS = 1e-2
    pose_kp = datum.poseKeypoints
    face_kp = datum.faceKeypoints
    hand_l_kp = datum.handKeypoints[0]
    hand_r_kp = datum.handKeypoints[1]
    max_area=0
    num_poses = len(pose_kp)
    for i in range(num_poses):

        filter_pose = pose_kp[i].copy()
        filter_pose = filter_pose[filter_pose[:, 2] > EPS]
        filter_pose = np.expand_dims(filter_pose, axis=0)
        x_min = min(filter_pose[0][:, 0])
        x_max = max(filter_pose[0][:, 0])
        y_min = min(filter_pose[0][:, 1])
        y_max = max(filter_pose[0][:, 1])

        # area
        area = (x_max - x_min) * (y_max - y_min)
        if area > max_area:
            max_area = area
            SELECTED_HUMAN_IDX = i

    pose_kp = np.array([pose_kp[SELECTED_HUMAN_IDX]])
    face_kp = np.array([face_kp[SELECTED_HUMAN_IDX]])
    hand_l_kp = np.array([hand_l_kp[SELECTED_HUMAN_IDX]])
    hand_r_kp = np.array([hand_r_kp[SELECTED_HUMAN_IDX]])

    return pose_kp, face_kp, hand_l_kp, hand_r_kp
    
def main(args):
    if args.img_dir[-1] != "/" or args.out_dir[-1] != "/":
        print("Try again. Path shoudl have / at the end")
        sys.exit(1)

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "models/"
    params["face"] = True
    params["hand"] = True
    params["write_json"] = osp.join(args.out_dir, "temp")
    os.makedirs(osp.join(args.out_dir, "temp"), exist_ok=True)
    assert len(os.listdir(osp.join(args.out_dir, "temp"))) == 0, "temp file not empty"
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # read images in dir
    s = time.time()
    print("start reading images ... ")
    IMGS = glob.glob(osp.join(args.img_dir, "**"), recursive=True)
    IMGS = sorted(
        [
            x
            for x in IMGS
            if x.split(".")[-1].lower() in ["png", "jpg", "jpeg", "bmp", "JPG"]
        ]
    )

    print("Done reading {} images after {} seconds".format(len(IMGS), time.time() - s))

    # run openpose for each iamge
    for i, IMG in tqdm(enumerate(IMGS), total=len(IMGS)):
        try:

            res = initialize_dict()
            IMGpath = osp.dirname(IMG.replace(args.img_dir, "")).strip(
                "/"
            )  # make sure slashes are removed
            IMGname = ".".join(osp.basename(IMG).split(".")[:-1])
            KPoutdir = osp.join(args.out_dir, "keypoints", IMGpath)
            IMGoutdir = osp.join(args.out_dir, "images", IMGpath)
            os.makedirs(IMGoutdir, exist_ok=True)
            os.makedirs(KPoutdir, exist_ok=True)

            # Process images
            datum = op.Datum()
            imageToProcess = cv2.imread(IMG)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            pose_kp, face_kp, hand_l_kp, hand_r_kp = process_keypoints(datum)

            # store keypoints in dictionary
            res["people"][0]["pose_keypoints_2d"] = pose_kp[0].flatten().tolist()
            res["people"][0]["face_keypoints_2d"] = face_kp[0].flatten().tolist()
            res["people"][0]["hand_left_keypoints_2d"] = hand_l_kp[0].flatten().tolist()
            res["people"][0]["hand_right_keypoints_2d"] = hand_r_kp[0].flatten().tolist()

            ## Save the keypoint results in json file
            with open(os.path.join(IMGoutdir, IMGname + ".json"), "w") as f:
                json.dump(res, f)
            
            # display if flag is True
            if args.display:
                cv2.imshow("Image", datum.cvOutputData)
                key = cv2.waitKey(15)
                if key == 27:
                    break
            if args.imgsave:
                print(osp.join(IMGoutdir, IMGname + ".png"))
                cv2.imwrite(osp.join(IMGoutdir, IMGname + ".png"), datum.cvOutputData)

            op_fn = glob.glob(osp.join(args.out_dir, "temp", "*.json"))
            assert len(op_fn) == 1, "Too many json files in folder."

            move_rename_kp(args, KPoutdir, op_fn[0], IMGname)

        except Exception as e:
            print(e, i, IMG)
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        default="../../examples/media/",
        help="dir with images, also in subfolders",
    )
    parser.add_argument(
        "--out_dir",
        default="../../examples/media/",
        help="dir where to write openpose files",
    )
    parser.add_argument(
        "--display",
        type=lambda x: x.lower() in ["True", "true", "1"],
        help="Enable to disable the visual display.",
    )
    parser.add_argument(
        "--imgsave",
        type=lambda x: x.lower() in ["True", "true", "1"],
        help="Save openpose iamge.",
    )
    args = parser.parse_args()
    main(args)
