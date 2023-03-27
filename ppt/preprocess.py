# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: preprocess.py
@Date: 2022/11/27 16:59
@Author: caijianfeng
"""
# In[2] dataset_arrow
import os

datasets_dir = './hdflowmind/'
dataset_split = os.listdir(datasets_dir)  # [train, val, test]
for dataset in dataset_split:
    dataset = os.path.join(datasets_dir, dataset)  # ./dataset_arrow/train
    dataset_ann_dir = os.path.join(dataset, 'annotations')  # ./dataset_arrow/train/annotations
    dataset_annotations_files = os.listdir(dataset_ann_dir)
    dataset_config = os.path.join(dataset, 'config.txt')  # ./dataset_arrow/train/config.txt
    with open(dataset_config, 'w') as f:
        for dataset_annotations_file in dataset_annotations_files:
            f.write(os.path.join(dataset_ann_dir, dataset_annotations_file) + '\n')
    print('dataset config file saving succeed!')
