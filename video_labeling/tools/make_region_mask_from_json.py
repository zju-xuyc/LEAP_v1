import os
import json
import numpy as np
import skimage.draw
import cv2
import argparse

parser = argparse.ArgumentParser("ByteTrack Demo!")
parser.add_argument("--videoname", type=str, default="jackson", help="video path")
parser.add_argument("--image_path", type=str, default="xxx", help="save video path")
parser.add_argument("--mask_foloder", type=str, default="./masks/", help="video name")
parser.add_argument("--path_annotation_json", type=str, default="xxx", help="video name")

args = parser.parse_args()

videoname = args.videoname
image_path = args.image_path
mask_foloder = args.mask_foloder
path_annotation_json = args.path_annotation_json

# 加载VIA导出的json文件
annotations = json.load(open(path_annotation_json, 'r'))
imgs = annotations["shapes"]
print(imgs)

# 图片路径
# 读出图片，目的是获取到宽高信息
image = cv2.imread(image_path)
height, width = image.shape[:2]
maskImage = np.zeros((height,width), dtype=np.uint8)

for imgId in imgs:
    regions = imgId
    print(regions)
    # 取出第一个标注的类别，本例只标注了一个物件
    polygons = regions['points']

    # 创建空的mask
    countOfPoints = len(polygons)
    points = [None] * countOfPoints
    for i, (x,y) in enumerate(polygons):
        points[i] = (int(x), int(y))

    contours = np.array(points)

    # 遍历图片所有坐标
    for i in range(width):
        for j in range(height):
            if cv2.pointPolygonTest(contours, (i, j), False) > 0:
                maskImage[j, i] = 255

savePath = mask_foloder + videoname + ".jpg"
# 保存mask
cv2.imwrite(savePath, maskImage)