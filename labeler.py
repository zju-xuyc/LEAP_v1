import cv2
import os
from settings.settings import video_details

def convert2traj(gt_labels):
    # [car_id,x1,y1,x2,y2,type]
    traj_dict = {}
    for detections in gt_labels:
        for detect in detections: 
            car_id = detect[0]
            detect = [int(x) for x in detect[1:-1]]            
            if car_id not in traj_dict:
                traj_dict[car_id] = [detect]
            else:
                traj_dict[car_id].append(detect)

    traj_list = []

    for key in traj_dict.keys():
        traj_list.append(traj_dict[key])

    return traj_list

def get_gt_labels(frame_list,video_save_path,video_name,frame_num):
    # 这里忽略了之前轨迹对应的标签关系
    traj_list = convert2traj(frame_list[:frame_num])
    threshlen = video_details.ThreshLen
    traj_filtered = []
    for traj in traj_list:
        if len(traj)>=threshlen:
            traj_filtered.append(traj)
    
    videoCapture = cv2.VideoCapture(os.path.join(video_save_path,video_name+".mp4"))
    success, background_img = videoCapture.read()
    
    return traj_filtered, background_img