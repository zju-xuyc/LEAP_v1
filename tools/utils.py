import cv2
import os
import tqdm
import numpy as np
from settings import settings
import copy
import cv2
import os
from settings.settings import video_details
from collections import Counter
import math
import time

def concat_videos(image_list_path,output_path,video_name,fps=25):
    
    image_list = os.listdir(image_list_path)
    image_list.sort(key = lambda x:int(x[-9:-4]))

    img = cv2.imread(os.path.join(image_list_path,image_list[0]))
    (height, width, _) = img.shape
    videoWriter = cv2.VideoWriter(os.path.join(output_path,video_name),\
        cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))

    for i in tqdm.trange(len(image_list)):
        img_path = os.path.join(image_list_path, image_list[i])
        img = cv2.imread(img_path)
        videoWriter.write(img)
    videoWriter.release()

def concat_video_slices(video_list_path,output_path,video_name):
    
    video_list = os.listdir(video_list_path)
    video_list.sort(key = lambda x:int(x[:-4]))
    video_list = video_list[:720]

    cap_settings = cv2.VideoCapture(os.path.join(video_list_path,video_list[0]))
    fps = cap_settings.get(cv2.CAP_PROP_FPS)
    width = int(cap_settings.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_settings.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_settings.release()
    videoWriter = cv2.VideoWriter(os.path.join(output_path,video_name),\
        cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
        # 支持的编码格式有限

    for i in tqdm.trange(len(video_list)):
        cap = cv2.VideoCapture(os.path.join(video_list_path,video_list[i]))
        while cap.isOpened():  
            ret, frame = cap.read()
            if ret:
                videoWriter.write(frame)
            else:
                break
        cap.release()
    videoWriter.release()


def calculate_IOU(rec1,rec2):
    """
    都是相对坐标的框，[x1,y1,x2,y2]
    计算IOU
    """
    rec1 = [int(x) for x in rec1]
    rec2 = [int(x) for x in rec2]
    left_max = max(rec1[0],rec2[0])
    top_max = max(rec1[1],rec2[1])
    right_min = min(rec1[2],rec2[2])
    bottom_min = min(rec1[3],rec2[3])
    
    if (left_max < right_min and bottom_min > top_max):
        rect1_area = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        rect2_area = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        area_cross = (bottom_min - top_max) * (right_min - left_max)
        return area_cross / (rect1_area + rect2_area - area_cross)

    return 0

def calculate_intersection(rec1,rec2):
    """_summary_
    Args:
        rec1 (bbox_list): detection
        rec2 (bbox_list): background_area
    """
    left_max = max(rec1[0],rec2[0])
    top_max = max(rec1[1],rec2[1])
    right_min = min(rec1[2],rec2[2])
    bottom_min = min(rec1[3],rec2[3])
    if (left_max < right_min and bottom_min > top_max):
        rect1_area = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        rect2_area = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        area_cross = (bottom_min - top_max) * (right_min - left_max)
        return area_cross / (rect1_area)
    return 0

def cosine_similarity(vector_a, vector_b):
    # 计算余弦相似度
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def pic_difference(vector_a, vector_b, pixel_num):
    diff = 0
    for i in range(256):
        diff += abs((vector_a[i][0]/pixel_num) - (vector_b[i][0]/pixel_num))
    return diff


def generate_video(video_path, video_save_path, video_name,fps=25):
    # 从图片生成视频
    file_list = os.listdir(video_path)
    file_list.sort(key=lambda x: int(x[-9:-4]))
    img_0 = cv2.imread(os.path.join(video_path,file_list[0]))
    
    size = (img_0.shape[1],img_0.shape[0])
    video = cv2.VideoWriter(os.path.join(video_save_path,video_name+".mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for file in file_list:
        image_path = os.path.join(video_path,file)
        img = cv2.imread(image_path)
        video.write(img)
        
    video.release()
    cv2.destroyAllWindows()

def convert_detection(detect):
    # input:  x1, y1, x2, y2, c
    # output: x_mid, y_mid, w, h, c
    p_w = video_details.patch_width
    x_mid = int((detect[0] + detect[2])/(2*p_w))
    y_mid = int((detect[1] + detect[3])/(2*p_w))
    
    width = int(detect[2] - detect[0])
    height = int(detect[3] - detect[1])

    return [x_mid, y_mid, width, height, float(detect[4])]

def matrix_fill(matrix2fill):
    # 矩阵缺失值填充
    for col in range(len(matrix2fill)):
        for row in range(len(matrix2fill[col])):
            if matrix2fill[col][row] == 0:
                matrix2fill[col][row] = [1,1]
    return matrix2fill

def get_img(image_base,frame_num):
    image_list = os.listdir(image_base)
    image_list.sort(key=lambda x: int(x[-9:-4]))
    image_name = os.path.join(image_base,image_list[frame_num])
    image = cv2.imread(image_name)
    return image

def associate_cars(detect1,detect2):
    # 根据IoU判断两辆车是不是同一辆
    # detect1 是当前帧中的所有检测 [id, x1,x2,y1,y2]
    # detect2 是最后一辆车的检测器检测结果 [x1,y1,x2,y2,confidence]
    detect_1 = detect1[1:]
    detect_2 =  detect2[:-1]
    iou = calculate_IOU(detect_1,detect_2)
    return iou

def match_last_car_with_gt(last_car,gt_detects,current_frame,cfg):
    # 用于将当前帧的检测和对应的标签对应上,用于后续的性能评估
    # gt_detects: [[id,x1,y1,x2,y2,object_type],[],[]] 不过要注意是str格式的，要转换
    # 添加了车辆信息之后，需要排除车辆类型信息
    last_car = last_car[:-1]
    iou_max = 0
    id_match = -1
    IoU_min = settings.IoU_min
    
    for gt_detect in gt_detects:
        gt_detect = gt_detect[:-1]
        gt_detect = [float(x) for x in gt_detect]
        last_car = [float(x) for x in last_car]
        iou = associate_cars(gt_detect,last_car)
        car_id = int(gt_detect[0])
        if iou > iou_max:
            if iou > IoU_min:
                id_match = car_id
            iou_max = iou
        
    if iou_max <= IoU_min: # 可能出现没有很好匹配的情况
        for i in range(max(cfg["start_frame"],current_frame-30), min(cfg["end_frame"],current_frame+30)):
            for gt_detect in video_details.gt_labels[i]:
                gt_detect = gt_detect[:-1]
                gt_detect = [float(x) for x in gt_detect]
                last_car = [float(x) for x in last_car]
                iou = associate_cars(gt_detect,last_car)
                car_id = int(gt_detect[0])
                if iou > iou_max:
                    if iou > IoU_min:
                        id_match = car_id
                    iou_max = iou

    if id_match==-1:
        print("检测车辆BBOX")
        print(last_car)
        print("当前帧的GT标签")
        print(gt_detects)
        id_match = list(video_details.gt_tuple.keys())[10]
    return id_match

def paint_image(img,gt_detects,detects):
    for gt_detect in gt_detects:
        car_id = int(gt_detect[0])
        cv2.putText(img,str(car_id)+" "+gt_detect[-1],(int(gt_detect[1])-10,int(gt_detect[2])-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.rectangle(img,(int(gt_detect[1]),int(gt_detect[2])),(int(gt_detect[3]),int(gt_detect[4])),(0,255,0),2)
    for detect in detects:
        class_num = int(detect[-1]) if int(detect[-1]) in settings.coco_names_invert.keys() else 0
        cv2.rectangle(img,(int(detect[0]),int(detect[1])),(int(detect[2]),int(detect[3])),(0,0,255),2)
        cv2.putText(img,settings.coco_names_invert[class_num],(int(detect[0])+20,int(detect[1])+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    return img


def locate_background_color(bg_color,clustered_color,perc):
    # 用于定位车辆颜色
    bg_color = np.array(bg_color)
    clustered_color = np.array(clustered_color)
    distance = 1e6
    filtered_color = 0
    index = -1
    # 首先确定和路面颜色最接近的颜色
    for i in range(3):
        if  perc[i] > 0.1:
            dis = sum((bg_color-clustered_color[i])**2)
            if dis < distance:
                distance = dis
                index = i
        
    index_2 = -1
    distance = 1e6

    for i in range(3):
        # 不可能是车辆颜色，直接忽略
        if perc[i] < 0.12:
            index_2 = i
            break
        if i != index:
            dis = sum((clustered_color[i]-np.array([0,0,0]))**2)
            if dis<distance:
                distance = dis
                index_2 = i
    for i in range(3):
        if i!=index and i!=index_2:
            filtered_color = clustered_color[i]
    return filtered_color

def get_background_color(img):

    img_shape = img.shape
    background_block = img[int(0.02*img_shape[0]):int(0.12*img_shape[0]),int(0.02*img_shape[1]):int(0.12*img_shape[1]),:]
    background_block = cv2.mean(background_block)[:3]
    background_block = [int(x) for x in background_block]
    return background_block

def palette_perc(k_cluster):

    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items(),reverse=False))

    colors = []
    for i in perc.keys():
        colors.append(list(k_cluster.cluster_centers_[i]))

    return colors,perc

def filter_tracks(tracks):
    # 使用阈值简单将轨迹分为有停顿和没有停顿的两类
    thresh_len = video_details.fps * 25
    min_len = video_details.fps * 2
    min_dis = video_details.v_height * 0.2
    track_filtered_by_length = [[],[]] 
    # 去除显著不合理的长轨迹
    for track in tracks:
        if len(track)<thresh_len and len(track)>min_len:
            if math.sqrt((track[0][0]-track[-1][0])*(track[0][0]-track[-1][0]) + \
                (track[0][1]-track[-1][1])*(track[0][1]-track[-1][1]))>min_dis:   
                track_start = [0.5*track[0][0]+0.5*track[0][2],0.5*track[0][1]+0.5*track[0][3]]
                track_end = [0.5*track[-1][0]+0.5*track[-1][2],0.5*track[-1][1]+0.5*track[-1][3]]
                flag = filter_by_region(track_start,track_end)       
                if flag:      
                    track_filtered_by_length[0].append(track)       
        
    return track_filtered_by_length

def judge_traffic_light(current_frame_id):
    traffic_light = settings.traffic_light[video_details.video_name]
    current_time = (current_frame_id - traffic_light[0][0]) % (traffic_light[1][0]+traffic_light[1][1])
    if current_time > traffic_light[1][0]:
        state = "red"
    else:
        state = "green"
    return state

def filter_by_region(start,end):
    # print("起始点")
    # print(start)
    # print(end)
    video_name = video_details.video_name
    in_out_region = settings.in_out_region[video_name]
    start_reg_id = -1
    end_reg_id = -1
    for id, item in enumerate(in_out_region):
        # id, [x1,y1,x2,y2] middle_x middle_y
        # print(item)
        if (start[0]>item[0] and start[0]<item[2]) and \
            (start[1]>item[1] and start[1]<item[3]):
                start_reg_id = id
        if (end[0]>item[0] and end[0]<item[2]) and \
            (end[1]>item[1] and end[1]<item[3]):
                end_reg_id = id
    # print(start_reg_id)
    # print(end_reg_id)           
    if start_reg_id != end_reg_id and start_reg_id!=-1 and end_reg_id!=-1:
        return True
    
    else:
        return False

def get_sample_gap(current_frame,tracks,match_traj_dict):

    # 按照分配的潜在路径，计算每个潜在路径的起始时间和结束时间
    
    interval = []
    dis_min = 1e4
    best_interval = []
    for key, value in match_traj_dict.items():
        track_id, nearest_point, distance = value
        estimated_starttime = current_frame - nearest_point
        estimated_endtime = current_frame + (len(tracks[track_id]) \
            - nearest_point)
        interval.append([estimated_starttime,estimated_endtime])
        if distance < dis_min:
            best_interval = [estimated_starttime,estimated_endtime]
            dis_min = distance

    return interval,best_interval

def justify_intersect(history_cache, curr_detect):
    # 上一帧对应的检测

    apply_traj = history_cache[-2]
    route_matched = {}
    closest_intersect = False

    for key_1 in apply_traj.keys():
        for key_2 in curr_detect.keys():
            if key_1 == key_2:
                route_matched[key_1] = [apply_traj[key_1],curr_detect[key_2]]

    dis = 1e4
    for key, value in route_matched.items():
        if value[0][-1] + value[1][-1] < dis:
            dis = value[0][-1] + value[1][-1]
            closest_intersect = [key,value]

    return closest_intersect
        
def justify_intersect_rate(interval_1,interval_2):
    # 判断两个区间重叠部分占两者并集的比例
    left_min = min(interval_1[0],interval_2[0])
    right_max = max(interval_1[1],interval_2[1])
    left_max = max(interval_1[0],interval_2[0])
    right_min = min(interval_1[1],interval_2[1])
    if right_min > left_max:        
        interset_ratio = abs(right_min-left_max)/abs(right_max-left_min)
        return interset_ratio
    else:
        return 0
    
def convert_float(results):
    tmp = []
    for result in results:
        result = [float(x) for x in result]
        tmp.append(result) 
    return tmp

    """_summary_
    这边的是用于Yolo模型的，之前因为文件重名导致的错误，所以这边的代码是没有用的，但是我还是保留了
    Returns:
        _type_: _description_
    """

import torch
import torchvision
import time

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im

def letterbox_first(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, new_unpad, shape, top, bottom, left, right

def letterbox_second(im, new_unpad, shape, top, bottom, left, right, color=(114, 114, 114)):

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 500  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

class Vehicle():
    def __init__(self, feature, frame, did, p_p, bbox):
        self.feature = feature
        self.frame_bound = [frame, frame]
        self.frame_list = [frame]
        self.did_list = [did]
        self.bbox_list = [p_p]
        self.bboxs = [bbox]
        self.update = False
    
    def add(self, feature, frame, did, p_p, bbox):
        self.feature = feature
        self.frame_bound[1] = frame
        self.frame_list.append(frame)
        self.did_list.append(did)
        self.bbox_list.append(p_p)
        self.bboxs.append(bbox)

def calculate_p_r_f(pred, targ):
    lt = max(pred[0], targ[0])
    rt = min(pred[1], targ[1])
    if rt - lt <= 0:
        return 0.0, 0.0, 0.0
    else:
        if pred[1] - pred[0] > 0:
            p = (rt - lt) / (pred[1] - pred[0])
            r = (rt - lt) / (targ[1] - targ[0])
            f = (2.0 * p * r) / (p + r)
            return p, r, f
        else:
            return 0, 0, 0

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    #  dist_mat.addmm_(1, -2, qf, gf.t())
    dist_mat.addmm_(qf,gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


if __name__ == "__main__":
    # concat_videos("/home/zju/xyc/video_data/M-30_mask",\
    #     "/home/zju/xyc/video_data/","M_30_mask.avi")
    concat_video_slices("/home/zju/xyc/otif/otif-dataset/raw_video/blazeit/taipei-2017-04-08",\
        "/home/zju/xyc/video_data/OTIF/blazeit/taipei","taipei_0408.mp4")