import re
import os
import cv2
import json
import itertools
import numpy as np
from glob import glob
import scipy.io as scio
from pycocotools import mask as cocomask
from PIL import Image
import glob
MAX_N = 10

categories = [
    {
        "supercategory": "none",
        "name": "master_chef_can",
        "id": 1
    },
    {
        "supercategory": "none",
        "name": "cracker_box",
        "id": 2
    },{
        "supercategory": "none",
        "name": "sugar_box",
        "id": 3
    },
    {
        "supercategory": "none",
        "name": "tomato_soup_can",
        "id": 4
    },
    {
        "supercategory": "none",
        "name": "mustard_bottle",
        "id": 5
    },
    {
        "supercategory": "none",
        "name": "tuna_fish_can",
        "id": 6
    },
    {
        "supercategory": "none",
        "name": "pudding_box",
        "id": 7
    },
    {
        "supercategory": "none",
        "name": "gelatin_box",
        "id": 8
    },
    {
        "supercategory": "none",
        "name": "potted_meat_can",
        "id": 9
    },
    {
        "supercategory": "none",
        "name": "banana",
        "id": 10
    },
    {
        "supercategory": "none",
        "name": "pitcher_base",
        "id": 11
    },
    {
        "supercategory": "none",
        "name": "bleach_cleanser",
        "id": 12
    },
    {
        "supercategory": "none",
        "name": "bowl",
        "id": 13
    },
    {
        "supercategory": "none",
        "name": "mug",
        "id": 14
    },
    {
        "supercategory": "none",
        "name": "power_drill",
        "id": 15
    },
    {
        "supercategory": "none",
        "name": "wood_block",
        "id": 16
    },
    {
        "supercategory": "none",
        "name": "scissors",
        "id": 17
    },
    {
        "supercategory": "none",
        "name": "large_marker",
        "id": 18
    },
    {
        "supercategory": "none",
        "name": "large_clamp",
        "id": 19
    },
    {
        "supercategory": "none",
        "name": "extra_large_clamp",
        "id": 20
    },
    {
        "supercategory": "none",
        "name": "foam_brick",
        "id": 21
    }

]


phases = ["train"]#, "test"]
for phase in phases:
    root_path = './'
    
    list_path = './' + phase + '_list.txt'
    #list_path = './' + phase + '.txt'
    classes = []
    for cls in open('./classes.txt'):
        if '\n' in cls:
            cls = cls[:-1]
        classes.append(cls)
    json_file = "{}.json".format(phase)

    
    

    res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    annot_count = 0
    image_id = 0
    processed = 0
    
    f = open(list_path)
    for line in f.readlines():
        
        if line[-1] == '\n':
            line = line[:-1]
        #line = 'data/' + line
        single_box_path = './' + line + '-box.txt'
        single_mat_path = './' + line + '-meta.mat'
        meta  = scio.loadmat(single_mat_path)
        meta['poses'] = meta['poses'].transpose(2,0,1)

        img_path = './' + line + '-color.png'

        img_elem = {"file_name": img_path,
                        "height": 480,
                        "width": 640,
                        "id": image_id}

        res_file["images"].append(img_elem)
        box_count = 0
        for box in open(single_box_path):
            
            cls_id = box.split(' ')[0]
            xmin = float(box.split(' ')[1])
            ymin = float(box.split(' ')[2])
            xmax = float(box.split(' ')[3])
            ymax = float(box.split(' ')[4])
            
            if xmax < xmin or ymax < ymin:
                continue
            w = xmax - xmin
            h = ymax - ymin
            poly = [[xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax]]
            annot_elem = {
                    "id": annot_count,
                    "bbox": [
                        float(xmin),
                        float(ymin),
                        float(w),
                        float(h)
                    ],
                    "RTs" : meta['poses'][box_count].tolist(),
                    "segmentation": list([poly]),
                    "image_id": image_id,
                    "ignore": 0,
                    "category_id": classes.index(cls_id) + 1,
                    "iscrowd": 0,
                    "area": float(w * h)
                }

            res_file["annotations"].append(annot_elem)
            annot_count += 1
            box_count +=1
        image_id += 1

        processed += 1

    with open(json_file, "w") as f:
        json_str = json.dumps(res_file)
        f.write(json_str)

    print("Processed {} {} images...".format(processed, phase))
print("Done.")