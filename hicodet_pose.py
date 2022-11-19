import json
import os
import numpy as np
import torch
from tqdm import tqdm
import torchvision.ops.boxes as box_ops

HUMAN_IDX = 49

basic_pose_json_path = "hicodet/basic_pose/basic_pose.json"

with open("hicodet/basic_pose/basic_pose.json") as f:
    basic_pose_json = json.load(f)

BASIC_POSE_JOINT = np.array(basic_pose_json[0]['joints'])
BASIC_POSE_JOINT_SCORE = np.zeros([17])

pose_path = "hicodet/detections/train_pose"
det_path = "hicodet/detections/train2015"
with_pose_det_path = "hicodet/detections/train2015_pose"

os.makedirs(with_pose_det_path, exist_ok=True)

det_json_list = os.listdir(det_path)

for det_json_path in tqdm(det_json_list):
    with open(os.path.join(pose_path, det_json_path)) as f:
        pose_json = json.load(f)    

    with open(os.path.join(det_path, det_json_path)) as f:
        det_json = json.load(f)
    
    for k,v in det_json.items():
        det_json[k] = np.array(v)

    if len(det_json['boxes']) == 0:
        assert False, "no det file"

    object_boxes = det_json['boxes'][det_json['labels']!=HUMAN_IDX]
    object_labels = det_json['labels'][det_json['labels']!=HUMAN_IDX]
    object_scores = det_json['scores'][det_json['labels']!=HUMAN_IDX]

    gt_human_boxes = det_json['boxes'][det_json['labels']==HUMAN_IDX]
    gt_human_labels = det_json['labels'][det_json['labels']==HUMAN_IDX]
    gt_human_scores = det_json['scores'][det_json['labels']==HUMAN_IDX]


    human_bbox_list = []
    human_label_list = []
    human_score_list = []
    human_joint_list = []
    human_joint_score_list = []
    for pose_det in pose_json:
        human_bbox_list.append(pose_det['bbox'])
        human_label_list.append(HUMAN_IDX)
        human_score_list.append(pose_det['score'])
        human_joint_list.append(pose_det['joints'])
        human_joint_score_list.append(pose_det['joints_score'])
    daram_human_boxes = np.array(human_bbox_list)
    daram_human_labels = np.array(human_label_list)
    daram_human_scores = np.array(human_score_list)
    daram_human_joints = np.array(human_joint_list)
    daram_human_joints_score= np.array(human_joint_score_list)
    
    if len(daram_human_boxes) == 0:
        nms_boxes = gt_human_boxes
        nms_scores = gt_human_scores
        active_scores = gt_human_scores
        nms_labels = gt_human_labels
        
    else:
        nms_boxes = np.concatenate([gt_human_boxes, daram_human_boxes])
        nms_scores = np.concatenate([gt_human_scores, daram_human_scores+1]) 
        active_scores = np.concatenate([gt_human_scores, daram_human_scores]) 
        nms_labels = np.concatenate([gt_human_labels, daram_human_labels])
    keep_idx = box_ops.batched_nms(torch.tensor(nms_boxes), torch.tensor(nms_scores), torch.tensor(nms_labels), 0.5).numpy()
      
    len_gt = len(gt_human_boxes)
    active_human_boxes = nms_boxes[keep_idx]
    active_human_labels = nms_labels[keep_idx]    
    active_human_scores = active_scores[keep_idx]
    
    active_human_joints = np.zeros([len(active_human_boxes), 34])
    active_human_joints_score = np.zeros([len(active_human_boxes), 17])
    
    gt_keep_idx = np.where(keep_idx<len_gt)[0]
    daram_keep_idx = np.where(keep_idx>=len_gt)[0]
    
    active_human_joints[gt_keep_idx] = BASIC_POSE_JOINT
    active_human_joints_score[gt_keep_idx] = BASIC_POSE_JOINT_SCORE
    
    daram_nms_idx = keep_idx[daram_keep_idx] - len_gt

    if len(daram_keep_idx) != 0:
        active_human_joints[daram_keep_idx] = daram_human_joints[daram_nms_idx]
        active_human_joints_score[daram_keep_idx] = daram_human_joints_score[daram_nms_idx]

    with_pose_json = {}
    with_pose_json['boxes'] = np.concatenate([active_human_boxes, object_boxes]).tolist()
    with_pose_json['labels'] = np.concatenate([active_human_labels, object_labels]).tolist()
    with_pose_json['scores'] = np.concatenate([active_human_scores, object_scores]).tolist()
    with_pose_json['human_joints'] = active_human_joints.tolist()
    with_pose_json['human_joints_score'] = active_human_joints_score.tolist()


