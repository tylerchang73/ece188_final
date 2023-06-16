# ece188
All checkpoint and image files are removed for space. To recreate:
- KITTI dataset is placed in data_kitti/ folder
- vod-converter-master is used to convert KITTI data to VOC format and placed in to data_voc/ folder
- mmdection is cloned from https://github.com/open-mmlab/mmdetection and placed in this repo.
- mmdetection_changes/kitti/ folder is moved to mmdetection/configs/
- mmdetection_changes/kitti_dataset.py file is moved to mmdetection/configs/_base_/datasets

Part 1: Signal Processing, Representation code is found in Project_Part1.ipynb
Part 2: Linear ALgebra, Reconstruction code is found in Project_Part2_Rect.ipynb and Project_Part2_Stereo.ipynb
Part 3: Deep Learning, Recognition code is found in Project_Part3_KITTI_Vis.ipynb, and through scripts found in Notes.txt
  Additional code/updates for Part 3 found in mmdetection_changes
