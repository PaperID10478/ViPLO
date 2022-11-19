import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import cv2


from torch.utils.data import Dataset
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import hflip
from torchvision.transforms import ColorJitter

from torch.utils.data.dataset import IterableDataset

# from vcoco.vcoco import VCOCO
from pycocotools.coco import COCO

from hicodet.hicodet import HICODet

from PIL import Image
import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, HandyTimer, BoxPairAssociation, all_gather
# from pocket.data import StandardTransform


def custom_collate(batch):
    images = []
    detections = []
    targets = []
    for im, det, tar in batch:
        images.append(im)
        detections.append(det)
        targets.append(tar)
    return images, detections, targets


class DataFactory(Dataset):
    def __init__(self,
            name, partition,
            data_root, detection_root,
            flip=False, color_jitter=False,
            box_score_thresh_h=0.2,
            box_score_thresh_o=0.2, backbone_name='resnet50', num_classes=117, pose=False
            ):
        
        self.pose = pose

        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)
        
        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}_vitpose.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict'), pose=pose
            )
            self.human_idx = 49
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}_vitpose.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            self.coco = COCO("v-coco/data/instances_vcoco_all_2014.json")

            self.human_idx = 1

        self.name = name
        self.detection_root = detection_root
        self.backbone_name = backbone_name
        
        self.box_score_thresh_h = box_score_thresh_h
        self.box_score_thresh_o = box_score_thresh_o
        self._flip = torch.randint(0, 2, (len(self.dataset),)) if flip \
            else torch.zeros(len(self.dataset))
        self._brightness = torch.randint(0, 2, (len(self.dataset),)) if color_jitter \
            else torch.zeros(len(self.dataset))
        self._contrast = torch.randint(0, 2, (len(self.dataset),)) if color_jitter \
            else torch.zeros(len(self.dataset))
        self._saturation = torch.randint(0, 2, (len(self.dataset),)) if color_jitter \
            else torch.zeros(len(self.dataset))
        self._hue = torch.randint(0, 2, (len(self.dataset),)) if color_jitter \
            else torch.zeros(len(self.dataset))

        self.aug_bri = ColorJitter(brightness=0.5)
        self.aug_con = ColorJitter(contrast=0.5)
        self.aug_sat = ColorJitter(saturation=0.5)
        self.aug_hue = ColorJitter(hue=0.3)

        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.dataset)

    def filter_detections(self, detection):
        """Perform NMS and remove low scoring examples"""

        boxes = torch.as_tensor(detection['boxes'])
        labels = torch.as_tensor(detection['labels'])
        scores = torch.as_tensor(detection['scores'])

        # Filter out low scoring human boxes
        idx = torch.nonzero(labels == self.human_idx).squeeze(1)
        keep_idx = idx[torch.nonzero(scores[idx] >= self.box_score_thresh_h).squeeze(1)]

        # Filter out low scoring object boxes
        idx = torch.nonzero(labels != self.human_idx).squeeze(1)
        keep_idx = torch.cat([
            keep_idx,
            idx[torch.nonzero(scores[idx] >= self.box_score_thresh_o).squeeze(1)]
        ])

        boxes = boxes[keep_idx].view(-1, 4)
        scores = scores[keep_idx].view(-1)
        labels = labels[keep_idx].view(-1)

        return dict(boxes=boxes, labels=labels, scores=scores)

    def flip_boxes(self, detection, target, w):
        detection['boxes'] = pocket.ops.horizontal_flip_boxes(w, detection['boxes'])
        
        if self.pose:
            human_joint = detection['human_joints']
            if len(human_joint) != 0:
                human_joint[:,:,0] = w - human_joint[:,:,0]
            detection['human_joints'] = human_joint
            
            human_joint = target['human_joints']
            if len(human_joint) != 0:
                human_joint[:,:,0] = w - human_joint[:,:,0]
            target['human_joints'] = human_joint

        target['boxes_h'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_h'])
        target['boxes_o'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_o'])

    def __getitem__(self, i):
        image, target = self.dataset[i]
        width, height = image.size

        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            if len(target['boxes_h'].shape) == 2:
                target['boxes_h'][:, :2] -= 1
            if len(target['boxes_o'].shape) == 2:
                target['boxes_o'][:, :2] -= 1
        else: ## v-coco
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')

        detection_path = os.path.join(
            self.detection_root,
            self.dataset.filename(i).replace('jpg', 'json')
        )

        with open(detection_path, 'r') as f:
            detection = json.load(f)
        
        if not self.pose:
            if 'human_joints' in detection.keys():
                detection.pop('human_joints')
                detection.pop('human_joints_score')        

        detection = pocket.ops.to_tensor(detection, input_format='dict')

        if self.pose:
            human_joint = detection['human_joints']
            if human_joint.dim() == 2:
                human_joint = human_joint.reshape(-1, 17, 2)
            
            detection['human_joints'] = human_joint

            human_joint = target['human_joints']
            if human_joint.dim() == 2:
                human_joint = human_joint.reshape(-1, 17, 2)
            target['human_joints'] = human_joint
  
        
        # random horizaontal flip
        if self._flip[i]:
            image = hflip(image)
            w, _ = image.size
            self.flip_boxes(detection, target, w)
        # random color jittering
        if self._brightness[i]:
            image = self.aug_bri(image)
        if self._contrast[i]:
            image = self.aug_con(image)
        if self._saturation[i]:
            image = self.aug_sat(image)
        if self._hue[i]:
            image = self.aug_hue(image)
        # random resize_crop 
        
        image = pocket.ops.to_tensor(image, 'pil')
        return image, detection, target

def sample(net, test_loader):
    result = {}
    net.eval()
    print("sample function start")
    for fid, batch in tqdm(enumerate(test_loader)):
        inputs = pocket.ops.relocate_to_cuda(batch[:-1])
        with torch.no_grad():
            output = net(*inputs)
        if output is None:
            continue

        assert len(output) == 1, "Batch size is not 1"
        output = pocket.ops.relocate_to_cpu(output[0])
        np_output = {}
        for key, value in output.items():
            if key not in ["prior", "weights", "phrase"]:
                np_output[key] = value.numpy()
        result[str(fid)] = np_output
        
    return result

def test(net, test_loader):
    testset = test_loader.dataset.dataset
    associate = BoxPairAssociation(min_iou=0.5)
    meter = DetectionAPMeter(
        600, nproc=1,
        num_gt=testset.anno_interaction,
        algorithm='11P'
    )
    net.eval()
    for batch in tqdm(test_loader):
        inputs = pocket.ops .relocate_to_cuda(batch[:-1])
        with torch.no_grad():
            output = net(*inputs)
        if output is None:
            continue

        # Batch size is fixed as 1 for inference
        assert len(output) == 1, "Batch size is not 1"
        output = pocket.ops.relocate_to_cpu(output[0])
        target = batch[-1][0]
        # Format detections
        box_idx = output['index'] # L
        boxes_h = output['boxes_h'][box_idx] # L x 4
        boxes_o = output['boxes_o'][box_idx] # L x 4
        objects = output['object'][box_idx] # L
        scores = output['scores'] # L
        verbs = output['prediction'] # L
        interactions = torch.tensor([
            testset.object_n_verb_to_interaction[o][v]
            for o, v in zip(objects, verbs)
        ]) #L
        # Associate detected pairs with ground truth pairs
        labels = torch.zeros_like(scores)
        unique_hoi = interactions.unique()
        for hoi_idx in unique_hoi:
            gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
            det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
            if len(gt_idx):
                labels[det_idx] = associate(
                    (target['boxes_h'][gt_idx].view(-1, 4),
                    target['boxes_o'][gt_idx].view(-1, 4)),
                    (boxes_h[det_idx].view(-1, 4),
                    boxes_o[det_idx].view(-1, 4)),
                    scores[det_idx].view(-1)
                )

        meter.append(scores, interactions, labels)

    return meter.eval()

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, train_loader, val_loader, num_classes=117, backbone_name='resnet-50', **kwargs):
        super().__init__(net, None, train_loader, **kwargs)
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.backbone_name = backbone_name
    def _on_start(self):
        self.meter = DetectionAPMeter(self.num_classes, algorithm='11P')
        self.hoi_loss = pocket.utils.SyncedNumericalMeter(maxlen=self._print_interval)
        self.intr_loss = pocket.utils.SyncedNumericalMeter(maxlen=self._print_interval)

    def _on_each_iteration(self):
        self._state.optimizer.zero_grad()
        output = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        loss_dict = output.pop()
        if loss_dict['hoi_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for lrank {self._rank}")
        
        self._state.loss = loss_dict['hoi_loss'] + loss_dict['interactiveness_loss']
        self._state.loss.backward()
        self._state.optimizer.step()
        self.hoi_loss.append(loss_dict['hoi_loss'])
        self.intr_loss.append(loss_dict['interactiveness_loss'])
           
        self._synchronise_and_log_results(output, self.meter)

    def _on_end_epoch(self):
        timer = HandyTimer(maxlen=2)
        # Compute training mAP
        if self._rank == 0:
            with timer:
                ap_train = self.meter.eval()
        # Run validation and compute mAP
        with timer:
            ap_val = self.validate()
        # Print performance and time
        if self._rank == 0:
            print("Epoch: {} | training mAP: {:.4f}, evaluation time: {:.2f}s |"
                "validation mAP: {:.4f}, total time: {:.2f}s\n".format(
                    self._state.epoch, ap_train.mean().item(), timer[0],
                    ap_val.mean().item(), timer[1]
            ))
            with open(os.path.join(self._cache_dir, 'log.txt'), "a") as f:
                f.write("Epoch: {} | training mAP: {:.4f}, evaluation time: {:.2f}s |"
                "validation mAP: {:.4f}, total time: {:.2f}s\n".format(
                    self._state.epoch, ap_train.mean().item(), timer[0],
                    ap_val.mean().item(), timer[1]
            ))
            self.meter.reset()
        #super()._on_end_epoch()
        if self._rank == 0:
            self.save_checkpoint()
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()
            
    def _print_statistics(self):
        super()._print_statistics()
        hoi_loss = self.hoi_loss.mean()
        intr_loss = self.intr_loss.mean()
        if self._rank == 0:
            print(f"=> HOI classification loss: {hoi_loss:.4f},",
            f"interactiveness loss: {intr_loss:.4f}")
            self.hoi_loss.reset()
            self.intr_loss.reset()

    def _synchronise_and_log_results(self, output, meter):
        scores = []; pred = []; labels = []
        # Collate results within the batch
        for result in output:
            scores.append(result['scores'].detach().cpu().numpy())
            pred.append(result['prediction'].cpu().float().numpy())
            labels.append(result["labels"].cpu().numpy())
        # Sync across subprocesses
        all_results = np.stack([
            np.concatenate(scores),
            np.concatenate(pred),
            np.concatenate(labels)
        ])
        all_results_sync = all_gather(all_results)
        # Collate and log results in master process
        if self._rank == 0:
            scores, pred, labels = torch.from_numpy(
                np.concatenate(all_results_sync, axis=1)
            ).unbind(0)
            meter.append(scores, pred, labels)

    @torch.no_grad()
    def validate(self):
        meter = DetectionAPMeter(self.num_classes, algorithm='11P')
        self._state.net.eval()
        for batch in self.val_loader:
            inputs = pocket.ops.relocate_to_cuda(batch)
            results = self._state.net(*inputs)
            self._synchronise_and_log_results(results, meter)

        # Evaluate mAP in master process
        if self._rank == 0:
            return meter.eval()
        else:
            return None
        