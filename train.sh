python cache.py --dataset vcoco --data-root vcoco --detection-dir vcoco/detections/test_upt --cache-dir vcoco_cache --partition test --model-path checkpoints/vcoco_clip_cls_upt/ckpt_01057_07.pt --cache-name vcoco_clip_cls_upt7.pkl 

python cache.py --dataset vcoco --data-root vcoco --detection-dir vcoco/detections/test_upt --cache-dir vcoco_cache --partition test --model-path checkpoints/vcoco_clip_cls_upt16/ckpt_01456_07.pt --patch-size 16 --cache-name vcoco_clip_cls_uptpatch16_7.pkl 

python cache.py --dataset vcoco --data-root vcoco --detection-dir vcoco/detections/test_upt --cache-dir vcoco_cache --partition test --model-path checkpoints/vcoco_clip_cls_upt16/ckpt_01664_08.pt --patch-size 16 --cache-name vcoco_clip_cls_uptpatch16_8.pkl 