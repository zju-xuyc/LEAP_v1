import cv2
import numpy as np
import os
import random

# /mnt/data_hdd_large/back_up/home/xyc/xyc/baseline_0226/feature_reid/datasets

dataset_name_dict={"jackson":"Jackson_Town","Jackson":"Jackson_Town","adventure":"Adventure_Rentals","Adventure":"Adventure_Rental",\
           "square":"Square_Northeast","Square":"Square_Northeast","taipei":"Taipei_Hires","Taipei":"Taipei_Hires","Caldot1":"caldot1",\
            "Caldot2":"caldot2","Shibuya":"shibuya"}

object_type = {2:'car',5:'bus',7:'truck'}
video_base = "/home/xuyanchao/video_data/Collected_live_videos/standard_split"
label_base = "/home/xuyanchao/video_data/pseudo_labels"
save_base = "/home/xuyanchao/fast_reid_train/datasets"
mask_base = "/home/xuyanchao/LEAP_v1/fixed_files/masks"

def get_labels(video_name,d_type):
    label_path = os.path.join(label_base,video_name,d_type,"label_%s_%s.txt"%(video_name,d_type))
    with open(label_path,"r") as f:
        contents = f.readlines()
    parsed_result = [[] for i in range(109000)]
    parsed_tuple = {}

    for content in contents:
        content = content.split(",")
        frame_id = int(content[0])
        car_id = int(content[1])
        x_min = max(0,int(float(content[2])))
        y_min = max(0,int(float(content[3])))
        x_max = x_min+int(float(content[4]))
        y_max = y_min+int(float(content[5]))
        conf = float(content[6])
        object_class = object_type[int(float(content[7]))]
        parsed_result[frame_id].append([car_id,x_min,y_min,x_max,y_max,conf,object_class])
        if car_id not in parsed_tuple.keys():
            parsed_tuple[car_id] = [frame_id,frame_id,object_class]
        else:
            parsed_tuple[car_id][1] = frame_id
    return parsed_result,parsed_tuple

def generate_train_dataset(video_name, skip_frame, parsed_result, parsed_tuple,fps=30,cam_id="9"):

    image_mask = cv2.imread(os.path.join(mask_base,video_name.lower()+"_mask.jpg"),cv2.IMREAD_GRAYSCALE)
    save_path = os.path.join(save_base,video_name,"image_train")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    video_path = os.path.join(video_base,dataset_name_dict[video_name],"concat","train",video_name+"_train.mp4")
    video_object = cv2.VideoCapture(video_path)
    print("Start cutting reid image")
    frame_count = 0

    while True:
        ret,frame = video_object.read()
        if ret:
            frame_masked = cv2.add(frame,np.zeros(np.shape(frame),dtype=np.uint8),mask=image_mask)
            labels = parsed_result[frame_count]
            if frame_count%skip_frame==0:
                for label in labels:
                    car_id,x_min,y_min,x_max,y_max,\
                        conf,object_class = label
                    # 过滤出现时间过短/长的车辆(或者说错误车辆)
                    duration_time = parsed_tuple[car_id][1] - parsed_tuple[car_id][0]
                    if duration_time<fps*30 and duration_time>fps*0.5:
                        pic = frame_masked[y_min:y_max,x_min:x_max,:]
                        cv2.imwrite(os.path.join(save_path,"%s_%d_%s_%d_%d_%d_%d_%d.jpg"%(cam_id,car_id,object_class,\
                            x_min,y_min,x_max,y_max,frame_count)),pic)
            frame_count += 1
        else:
            break

def generate_test_query_dataset(video_name, skip_frame, parsed_result, parsed_tuple,fps=30,cam_id="9"):

    image_mask = cv2.imread(os.path.join(mask_base,video_name.lower()+"_mask.jpg"),cv2.IMREAD_GRAYSCALE)
    save_path_test = os.path.join(save_base,video_name,"image_test")

    if not os.path.exists(save_path_test):
        os.mkdir(save_path_test)

    save_path_query = os.path.join(save_base,video_name,"image_query")
    if not os.path.exists(save_path_query):
        os.mkdir(save_path_query)

    video_path = os.path.join(video_base,dataset_name_dict[video_name],"concat","test",video_name+"_test.mp4")
    video_object = cv2.VideoCapture(video_path)
    print("Start cutting reid image")
    frame_count = 0

    test_dict = {}

    while True:
        ret,frame = video_object.read()
        if ret:
            frame_masked = cv2.add(frame,np.zeros(np.shape(frame),dtype=np.uint8),mask=image_mask)
            labels = parsed_result[frame_count]
            if frame_count%skip_frame==0:
                for label in labels:
                    car_id,x_min,y_min,x_max,y_max,\
                        conf,object_class = label
                    # 过滤出现时间过短/长的车辆(或者说错误车辆)
                    duration_time = parsed_tuple[car_id][1] - parsed_tuple[car_id][0]
                    if duration_time<fps*30 and duration_time>fps*0.5:
                        pic = frame_masked[y_min:y_max,x_min:x_max,:]
                        if car_id not in test_dict.keys():
                            test_dict[car_id] = [[pic,object_class,x_min,y_min,x_max,y_max,frame_count]]
                        else:
                            test_dict[car_id].append([pic,object_class,x_min,y_min,x_max,y_max,frame_count])
            frame_count += 1
        else:

            for key in test_dict.keys():
                query_id = random.randint(0,len(test_dict[key])-1)
                for i in range(len(test_dict[key])):
                    if i != query_id:
                        pic,object_class,x_min,y_min,x_max,y_max,frame_count = test_dict[key][i]
                        cv2.imwrite(os.path.join(save_path_test,"%s_%d_%s_%d_%d_%d_%d_%d.jpg"%(cam_id,key,object_class,\
                            x_min,y_min,x_max,y_max,frame_count)),pic)
                    else:
                        pic,object_class,x_min,y_min,x_max,y_max,frame_count = test_dict[key][i]
                        cv2.imwrite(os.path.join(save_path_query,"%s_%d_%s_%d_%d_%d_%d_%d.jpg"%(cam_id,key,object_class,\
                            x_min,y_min,x_max,y_max,frame_count)),pic)
            break

def get_object_counts_duration(dataset_name,d_type="test",fps=30):

    parsed_result,parsed_tuple = get_labels(dataset_name,d_type=d_type)

    time_dict = {"car":0,"bus":0,"truck":0}
    # Calculate Occupency
    count = 0
    for contents in parsed_result:

        count += 1
        if len(contents) > 0:
            count_max = count
        for content in contents:
            if content[-1] == "car":
                time_dict["car"] += 1
                continue

    for contents in parsed_result:
        for content in contents:
            if content[-1] == "bus":
                time_dict["bus"] += 1
                continue

    for contents in parsed_result:
        for content in contents:
            if content[-1] == "truck":
                time_dict["truck"] += 1
                continue
    
    object_occupency = {key: value / count_max for key, value in time_dict.items()}

    object_count = {"car":0,"bus":0,"truck":0}
    object_duration = {"car":0,"bus":0,"truck":0}

    for object_id in parsed_tuple.keys():
        duration_time = parsed_tuple[object_id][1] - parsed_tuple[object_id][0]
        if duration_time<fps*30 and duration_time > fps*0.5:
            object_count[parsed_tuple[object_id[-1]]] += 1
            object_duration[parsed_tuple[object_id[-1]]] += duration_time

    return object_occupency, object_count, object_duration

if __name__ == "__main__":
    # 9,10,11
    parsed_result,parsed_tuple = get_labels("jackson",d_type="train")

    generate_train_dataset("jackson",15,parsed_result,parsed_tuple,cam_id="1")

    parsed_result,parsed_tuple = get_labels("jackson",d_type="test")

    generate_test_query_dataset("jackson",15,parsed_result,parsed_tuple,cam_id="1")