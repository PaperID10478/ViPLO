
python test.py --detection-dir hicodet/detections/test2015_upt_vitpose --model-path checkpoints/clip_cls_117_final16_gamma03/ckpt_10983_07.pt --warp --pose --local_pose --patch-size 16

python test.py --detection-dir hicodet/detections/test2015_gt_vitpose --model-path checkpoints/clip_cls_117_final16_gamma03/ckpt_10983_07.pt --warp --pose --local_pose --patch-size 16

python test.py --detection-dir hicodet/detections/test2015_gt_vitpose --model-path checkpoints/clip_cls_117_final32_gamma03/ckpt_07987_07.pt --warp --pose --local_pose --patch-size 32