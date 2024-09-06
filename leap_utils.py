import yaml
import torch
import numpy as np
import random
import os
import shutil
import cv2
from settings.settings import video_details
from tools.utils import calculate_IOU
import math

def getYaml(file_path):
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def set_seeds(seed):
    # 设置全局随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def clear_dir(cfgs):
    # 清空文件夹
    if not os.path.exists('./outputs/selected_frames'):
        os.mkdir('./outputs/selected_frames')
    else:
        shutil.rmtree('./outputs/selected_frames')
        os.mkdir('./outputs/selected_frames')

    if not os.path.exists('./outputs/selected_frames_origin'):
        os.mkdir('./outputs/selected_frames_origin')
    else:
        shutil.rmtree('./outputs/selected_frames_origin')
        os.mkdir('./outputs/selected_frames_origin')

    if not os.path.exists('./outputs/traj_compute_tmp'):
        os.mkdir('./outputs/traj_compute_tmp')
    else:
        shutil.rmtree('./outputs/traj_compute_tmp')
        os.mkdir('./outputs/traj_compute_tmp')

    if not os.path.exists('./outputs/reid/%s'%(cfgs["video_name"])):
        os.mkdir('./outputs/reid/%s'%(cfgs["video_name"]))
    else:
        shutil.rmtree('./outputs/reid/%s'%(cfgs["video_name"]))
        os.mkdir('./outputs/reid/%s'%(cfgs["video_name"]))

    if not os.path.exists('./outputs/traj_match'):
        os.mkdir('./outputs/traj_match')
    else:
        shutil.rmtree('./outputs/traj_match')
        os.mkdir('./outputs/traj_match')

    if not os.path.exists('./outputs/parsed_results'):
        os.mkdir('./outputs/parsed_results')
    
    if not os.path.exists('./outputs/traj_saved'):
        os.mkdir('./outputs/parsed_results')

"""
衡量轨迹的质量，包括可视化以及相关指标
"""

def paint_traj(paint_path, tracks):
    """
    visualize the predicted trajectory and the ground truth trajectory
    """
    gt_tuple = {}
    for labels in video_details.gt_labels:
        for label in labels:
            if label[0] not in gt_tuple.keys():
                gt_tuple[label[0]] = [label[1:5]]
            else:
                gt_tuple[label[0]].append(label[1:5])   
               
    for key in video_details.return_tuple.keys():
        bg_img = video_details.background_img.copy()
        traj_id = video_details.return_tuple[key][-2]
        gt_id = video_details.match_dict[key]
        pred_traj = tracks[0][traj_id]
        gt_traj = gt_tuple[gt_id]
        for point in pred_traj:
            cv2.circle(bg_img,(int(0.5*(point[0]+point[2])),int(0.5*(point[1]+point[3]))),1,(0,0,255),2)
            
        cv2.rectangle(bg_img,(int(pred_traj[0][0]),int(pred_traj[0][1])),(int(pred_traj[0][2]),int(pred_traj[0][3])),(0,0,255),2)
        cv2.rectangle(bg_img,(int(pred_traj[-1][0]),int(pred_traj[-1][1])),(int(pred_traj[-1][2]),int(pred_traj[-1][3])),(0,0,255),2)
        
        for point in gt_traj:
            cv2.circle(bg_img,(int(0.5*(point[0]+point[2])),int(0.5*(point[1]+point[3]))),1,(0,255,0),2)
            
        cv2.rectangle(bg_img,(int(gt_traj[0][0]),int(gt_traj[0][1])),(int(gt_traj[0][2]),int(gt_traj[0][3])),(0,255,0),2)
        cv2.rectangle(bg_img,(int(gt_traj[-1][0]),int(gt_traj[-1][1])),(int(gt_traj[-1][2]),int(gt_traj[-1][3])),(0,255,0),2)
        
        cv2.imwrite(os.path.join(paint_path,"%d.jpg"%(key)),bg_img)


def get_estimate_end_point(pred_traj,time_sampled,bbox_sampled):
    
    s_w = pred_traj[0][2]-pred_traj[0][0]
    s_h = pred_traj[0][3]-pred_traj[0][1]
    e_w = pred_traj[-1][2]-pred_traj[-1][0]
    e_h = pred_traj[-1][3]-pred_traj[-1][1]
    m_w = pred_traj[time_sampled][2]-pred_traj[time_sampled][0]
    m_h = pred_traj[time_sampled][3]-pred_traj[time_sampled][1]
    actual_w = bbox_sampled[2]-bbox_sampled[0]
    actual_h = bbox_sampled[3]-bbox_sampled[1]
    s_ratio_w = s_w/m_w
    s_ratio_h = s_h/m_h
    e_ratio_w = e_w/m_w
    e_ratio_h = e_h/m_h
    start_point = (0.5*(pred_traj[0][0]+pred_traj[0][2]), 0.5*(pred_traj[0][1]+pred_traj[0][3]))
    end_point = (0.5*(pred_traj[-1][0]+pred_traj[-1][2]), 0.5*(pred_traj[-1][1]+pred_traj[-1][3]))
    estimate_start_bbox = [start_point[0]-0.5*actual_w*s_ratio_w,start_point[1]-0.5*actual_h*s_ratio_h,\
        start_point[0]+0.5*actual_w*s_ratio_w,start_point[1]+0.5*actual_h*s_ratio_h]
    estimate_end_bbox = [end_point[0]-0.5*actual_w*e_ratio_w,end_point[1]-0.5*actual_h*e_ratio_h,\
        end_point[0]+0.5*actual_w*e_ratio_w,end_point[1]+0.5*actual_h*e_ratio_h]
    
    return estimate_start_bbox, estimate_end_bbox
 
def evaluate_traj_acc(tracks,ignore_direction=True,threshold=200):
    """
    Evaluate whether the prediction is correct by the start/end point of the trajectory
    """    
    ignore_direction = ignore_direction
    gt_tuple = {}
    
    count = 0
    acc_count = 0
    
    for labels in video_details.gt_labels:
        for label in labels:
            if label[0] not in gt_tuple.keys():
                gt_tuple[label[0]] = [label[1:5]]
            else:
                gt_tuple[label[0]].append(label[1:5])
                
    for key in video_details.return_tuple.keys():
        # bg_img = video_details.background_img.copy()
        traj_id = video_details.return_tuple[key][-2]
        gt_id = video_details.match_dict[key]
        pred_traj = tracks[0][traj_id]
        gt_traj = gt_tuple[gt_id]
        gt_start = (int(0.5*(gt_traj[0][0]+gt_traj[0][2])),int(0.5*(gt_traj[0][1]+gt_traj[0][3])))
        gt_end = (int(0.5*(gt_traj[-1][0]+gt_traj[-1][2])),int(0.5*(gt_traj[-1][1]+gt_traj[-1][3])))
        pred_start = (int(0.5*(pred_traj[0][0]+pred_traj[0][2])),int(0.5*(pred_traj[0][1]+pred_traj[0][3])))
        pred_end = (int(0.5*(pred_traj[-1][0]+pred_traj[-1][2])),int(0.5*(pred_traj[-1][1]+pred_traj[-1][3])))
        
        start_point_err = math.sqrt((gt_start[0]-pred_start[0])**2+(gt_start[1]-pred_start[1])**2)
        end_point_err = math.sqrt((gt_end[0]-pred_end[0])**2+(gt_end[1]-pred_end[1])**2)
        
        start_end_err = math.sqrt((gt_start[0]-pred_end[0])**2+(gt_start[1]-pred_end[1])**2)
        end_start_err = math.sqrt((gt_end[0]-pred_start[0])**2+(gt_end[1]-pred_start[1])**2)
        
        count += 2
        if ignore_direction:
            if start_point_err < threshold or start_end_err < threshold:
                acc_count += 1
            if end_point_err < threshold or end_start_err < threshold:
                acc_count += 1
        else: 
            if start_end_err < threshold:
                acc_count += 1
            if end_start_err < threshold:
                acc_count += 1
                
    return acc_count/count    

def evaluate_traj(paint_path, tracks):
    # 记录和真实轨迹开始和结束的位置误差 bbox的误差
    start_count = 0
    correct_start = 0
    correct_end = 0
    start_iou = []
    end_iou = []
    
    gt_tuple = {}
    for labels in video_details.gt_labels:
        for label in labels:
            if label[0] not in gt_tuple.keys():
                gt_tuple[label[0]] = [label[1:5]]
            else:
                gt_tuple[label[0]].append(label[1:5])
    
    for key in video_details.return_tuple.keys():
        # 后续的改进可以把实际采样的一帧和预测以及结束的一帧可视化
        start_count+=1
        bg_img = video_details.background_img.copy()
        traj_id = video_details.return_tuple[key][-2]
        gt_id = video_details.match_dict[key]
        pred_traj = tracks[0][traj_id]
        gt_traj = gt_tuple[gt_id]
        time_sampled = video_details.bbox_rate[key][-1]
        bbox_sampled = video_details.bbox_rate[key][:4]
        estimate_start_bbox, estimate_end_bbox = get_estimate_end_point(pred_traj,time_sampled,bbox_sampled)
        start_overlap = calculate_IOU(gt_traj[0][:4],estimate_start_bbox)
        end_overlap = calculate_IOU(gt_traj[-1][:4],estimate_end_bbox)

        if start_overlap>0:
            correct_start += 1 
        if end_overlap>0:
            correct_end += 1 
        start_iou.append(start_overlap)
        end_iou.append(end_overlap)
    
        estimate_start_bbox = [int(i) for i in estimate_start_bbox]
        estimate_end_bbox = [int(i) for i in estimate_end_bbox]
        cv2.rectangle(bg_img,(estimate_start_bbox[0],estimate_start_bbox[1]),\
            (estimate_start_bbox[2],estimate_start_bbox[3]),(0,255,0),2)
        cv2.rectangle(bg_img,(estimate_end_bbox[0],estimate_end_bbox[1]),\
            (estimate_end_bbox[2],estimate_end_bbox[3]),(0,255,0),2)
        
        cv2.rectangle(bg_img,(gt_traj[0][0],gt_traj[0][1]),\
            (gt_traj[0][2],gt_traj[0][3]),(0,0,255),2)
        cv2.rectangle(bg_img,(gt_traj[-1][0],gt_traj[-1][1]),\
            (gt_traj[-1][2],gt_traj[-1][3]),(0,0,255),2)
        cv2.imwrite(os.path.join(paint_path,"%d_bbox.jpg"%(key)),bg_img)
        
    print("Accuracy of the start point")
    print(correct_start/start_count)
    print("Accuracy of the end point")
    print(correct_end/start_count)
    print("Start point IoU")
    print(np.mean(start_iou))
    print("End point IoU")
    print(np.mean(end_iou))
        

