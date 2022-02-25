# coding=utf-8
import os
import cv2
import sys
import json
import numpy as np
import shutil


def conver(dataclass='train'):
    dataset = {"info": {
        "description": "WIDER face in COCO format.",
        "url": "",
        "version": "1.1",
        "contributor": "tianxingxia",
        "date_created": "2022-02-22"},
        "images": [],
        "annotations": [],
        "categories": [{"supercategory": "none", "id": 1, "name": "face"}],
    }

    outputpath = ""
    image_root = 'WIDER_' + dataclass + '/images/'
    phase = "WIDERFace" + dataclass.capitalize() + "COCO"

    with open('wider_face_split/wider_face_' + dataclass + '_bbx_gt.txt', 'r') as f:
        lines = f.readlines()
        num_lines = len(lines)
        i_l = 0
        img_id = 1
        anno_id = 1
        imagepath = None
        while i_l < num_lines:
            # print(num_lines, '\\', i_l, '-', img_id)
            if len(lines[i_l]) < 1:
                break
            if '--' in lines[i_l]:
                imagepath = lines[i_l].strip()
                im = image_root + imagepath
                if os.path.exists(im):
                    im = cv2.imread(im)
                    height, width, channels = im.shape
                    dataset["images"].append(
                        {"file_name": imagepath, "coco_url": "local", "height": height, "width": width,
                         "flickr_url": "local", "id": img_id})
                    i_l += 1
                    num_gt = int(lines[i_l])
                    while num_gt > 0:
                        i_l += 1
                        x1, y1, wid, hei = list(map(int, lines[i_l].split()))[:4]
                        num_gt -= 1

                        if wid <= 0 or hei <= 0:
                            print(f'图像id:{img_id}有无效标注:x1={x1},wid={wid},y1={y1},hei={hei}')

                        else:
                            dataset["annotations"].append({
                                "segmentation": [],
                                "iscrowd": 0,
                                "area": wid * hei,
                                "image_id": img_id,
                                "bbox": [x1, y1, wid, hei],
                                "category_id": 1,
                                "id": anno_id})
                            anno_id = anno_id + 1

                    img_id += 1
                else:
                    i_l += 1
            i_l += 1

    json_name = os.path.join(outputpath, "{}.json".format(phase))

    with open(json_name, 'w') as f:
        json.dump(dataset, f)


print('开始转换,并清除无效标注...')
conver('train')
print('训练集转换完毕')
conver('val')
print('验证集转换完毕')

