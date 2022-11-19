import json
import os
from re import L
import numpy as np
import torch
from tqdm import tqdm
import torchvision.ops.boxes as box_ops

HUMAN_IDX = 49
IOU_THRESHOLD = 0.5

basic_pose_json_path = "hicodet/basic_pose/basic_pose.json"

with open("hicodet/basic_pose/basic_pose.json") as f:
    basic_pose_json = json.load(f)

BASIC_POSE_JOINT = torch.tensor(basic_pose_json[0]['joints'])
BASIC_POSE_JOINT_SCORE = torch.zeros([17])

pose_path = "hicodet/detections/train_pose"

instance_json_path = "hicodet/instances_train2015.json"
instance_save_json_path = "hicodet/instances_train2015_pose.json"

with open(instance_json_path) as f:
    instance_json = json.load(f)

annos = instance_json['annotation']
filenames = instance_json['filenames']

for anno, filename in tqdm(zip(annos, filenames)):
    with open(os.path.join(pose_path, filename.replace('jpg', 'json'))) as f:
        pose_json = json.load(f)

    joints_result = []
    joints_score_result = []
    for box_h in anno['boxes_h']:
        with_pose_box = torch.tensor(list(map(lambda x: x['bbox'], pose_json)))
        joints = torch.tensor(list(map(lambda x: x['joints'], pose_json)))
        joints_score = torch.tensor(list(map(lambda x: x['joints_score'], pose_json)))

        if len(pose_json) == 0:
            joints_result.append(BASIC_POSE_JOINT)
            joints_score_result.append(BASIC_POSE_JOINT_SCORE)
            continue
    
        iou_result = box_ops.box_iou(torch.tensor(box_h).unsqueeze(0), with_pose_box).squeeze(0)
        need_idx = torch.nonzero(iou_result>IOU_THRESHOLD).squeeze(1)
        if len(need_idx) == 1:
            joints_result.append(joints[need_idx.item()])
            joints_score_result.append(joints_score[need_idx.item()])
        elif len(need_idx) == 0:
            joints_result.append(BASIC_POSE_JOINT)
            joints_score_result.append(BASIC_POSE_JOINT_SCORE)
        else:
            need_idx = torch.argmax(iou_result[need_idx])

            joints_result.append(joints[need_idx.item()])
            joints_score_result.append(joints_score[need_idx.item()])
    
    if len(joints_result) > 0:
        joints_result = torch.stack(joints_result)
        joints_score_result = torch.stack(joints_score_result)
    else:
        joints_result = torch.tensor([])
        joints_score_result = torch.tensor([])

    anno['human_joints'] = joints_result.tolist()
    anno['human_joints_score'] = joints_score_result.tolist()

with open(instance_save_json_path, "w") as f:
    json.dump(instance_json, f)

