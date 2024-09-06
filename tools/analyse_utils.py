import cv2
import os
import numpy as np
import json
from settings import settings
import xml.dom.minidom

coco_names_invert = {7:"truck",2:"car",5:"bus"}

def paint_tracks(video_name,tracks, desc="all"):
    # 绘制不同场景下的轨迹
    img_background = cv2.imread("./fixed_files/masks/background/%s.jpg"%(video_name))
    colors = [(255,255,255),(0,0,0),(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,100,255),(0,255,255)]
    count = 0
    for track in tracks:
        count += 1
        for point in track:
            cv2.circle(img_background,(int(0.5*(point[0]+point[2])),int(0.5*(point[1]+point[3]))),1,colors[count%8],2)
    cv2.imwrite("./fixed_files/masks/annotated_region/%s_tracks_%s.jpg"%(video_name,desc),img_background)
    
def paint_tracks_with_id(video_name, desc="part"):
    label_parsed, tuple_dict = get_blazeit_labels(video_name)
    tracks_list = []
    if desc == "all":
        tracks = {}
        for labels in label_parsed:
            for label in labels:
                if label[0] not in tracks.keys():
                    tracks[label[0]] = []
                tracks[label[0]].append(label[1:-1])
        for key,item in tracks.items():
            tracks_list.append(item)
    else:
        tracks = np.load("./fixed_files/preprocessed/%s/%s_0_7200_tracks_filtered.npy"%(video_name,video_name),\
            allow_pickle=True)
        tracks = tracks[0]
        for i in range(len(tracks)):
            tracks_list.append([])
            for t in tracks[i]:
                tracks_list[-1].append(t[:-2])
    paint_tracks(video_name, tracks_list, desc)

def get_blazeit_labels(video_name):
    """
    临时使用,后续已整合到data_prepare中
    """
    label_file = os.path.join(settings.LABEL_BASE,video_name,"test",\
                "label_%s_test.txt"%(video_name))

    with open(label_file,"r") as f:
        pseudo_label = f.readlines()
    f.close()
    label_parsed = [[] for i in range(111300)]
    tuple_dict = {}

    for label in pseudo_label:

        label = label.split(",")
        frame_id = int(label[0])
        car_id = int(label[1])
        x_min = int(float(label[2]))
        y_min = int(float(label[3]))
        x_max = x_min+int(float(label[4]))
        y_max = y_min+int(float(label[5]))
        obj_class = int(float(label[7]))
        object_name = coco_names_invert[obj_class]
        label_parsed[frame_id].append([car_id,x_min,y_min,x_max,y_max,object_name])
        if car_id not in tuple_dict.keys():
            tuple_dict[car_id] = [frame_id,frame_id,[x_min,y_min,x_max,y_max],[x_min, y_min, x_max, y_max],[]]
        else:
            tuple_dict[car_id][1] = frame_id
            tuple_dict[car_id][3] = [x_min, y_min, x_max, y_max]
        tuple_dict[car_id][-1].append(obj_class)
    
    car_ids = list(tuple_dict.keys())
    tuple_dict_origin = tuple_dict.copy()
    for car_id in car_ids:
        if abs(tuple_dict[car_id][1] - tuple_dict[car_id][0]) < 30:
            del tuple_dict[car_id]
            continue
        
        start_xmin, start_ymin, start_xmax, start_ymax = tuple_dict[car_id][2]
        end_xmin,   end_ymin,   end_xmax,   end_ymax   = tuple_dict[car_id][3]
        start_xcenter, start_ycenter = int((start_xmin + start_xmax)/2.0), int((start_ymin + start_ymax)/2.0)
        end_xcenter,   end_ycenter = int((end_xmin + end_xmax)/2.0), int((end_ymin + end_ymax)/2.0)
        
        if ((start_xcenter - end_xcenter)**2+(start_ycenter - end_ycenter)**2)**0.5 < 20:
        # 过短的轨迹不予考虑
            del tuple_dict[car_id]
            continue
        
        class_number_dict = {}
        for class_id in tuple_dict[car_id][-1]:
            if class_id not in class_number_dict:
                class_number_dict[class_id] = 0
            class_number_dict[class_id] += 1
        all_class_numbers = []
        for class_id, class_number in class_number_dict.items():
            all_class_numbers.append([class_id, class_number])
        tuple_dict[car_id][-1] = max(all_class_numbers, key=lambda x:x[1])[0]
    return label_parsed, tuple_dict

def get_m30_labels(video_name,K):
    coco_names = {"car":2,"bus":5,"truck":7,0:"others","van":7,"big-truck":7}
    label_files = os.listdir(os.path.join(settings.VIDEO_BASE+"/M30/annotations",video_name,"xml"))
    label_files.sort(key=lambda x:int(x.split(".")[0]))
    label_files = label_files[:K]
    frame_num_all = K
    label_parsed = [[] for i in range(frame_num_all)]
    tuple_dict = {}

    for file in label_files:
        frame_num = int(file.split(".")[0])
        dom = xml.dom.minidom.parse(os.path.join(settings.VIDEO_BASE+"/M30/annotations",video_name,"xml",file))
        root = dom.documentElement
        objects = root.getElementsByTagName("object")

        for object in objects:
            object_class = object.getElementsByTagName("class")[0].childNodes[0].nodeValue

            if object_class == "car" or object_class == "truck" or object_class == "big-truck" or object_class == "van":
                car_id = int(object.getElementsByTagName("ID")[0].childNodes[0].nodeValue)   
                bnd_box = object.getElementsByTagName("bndbox")[0]
                xmin = int(bnd_box.getElementsByTagName("xmin")[0].childNodes[0].nodeValue) 
                ymin = int(bnd_box.getElementsByTagName("ymin")[0].childNodes[0].nodeValue)
                xmax = int(bnd_box.getElementsByTagName("xmax")[0].childNodes[0].nodeValue)
                ymax = int(bnd_box.getElementsByTagName("ymax")[0].childNodes[0].nodeValue)   
                label_parsed[frame_num].append([car_id,xmin,ymin,xmax,ymax,object_class])   # Frame number need attention 

                if car_id not in tuple_dict.keys():
                    tuple_dict[car_id] = [frame_num,frame_num,[[xmin,ymin,xmax,ymax]],coco_names[object_class]]
                else:
                    tuple_dict[car_id][1] = frame_num
                    tuple_dict[car_id][2].append([xmin, ymin, xmax, ymax])

    return  label_parsed, tuple_dict

def extract_frame_from_video(path,k,save_name):
    
    video = cv2.VideoCapture(path)
    video.set(cv2.CAP_PROP_POS_FRAMES, k)
    _, frame = video.read()
    cv2.imwrite(save_name + '.jpg', frame)

def LRE_algorithm(result_tuple):
    # 理论最优采样算法
    vehicle_range = []
    sampled_frames = []
    for key, value in result_tuple.items():
        vehicle_range.append([result_tuple[key][0],result_tuple[key][1]])
    vehicle_range = sorted(vehicle_range, key=lambda x:x[1])
    sampled_frames.extend([vehicle_range[0][0]])
    current_select = vehicle_range[0][0]
    while True:
        for i in range(len(vehicle_range)):
            if vehicle_range[i][0] > current_select:
                current_select = vehicle_range[i][1]
                sampled_frames.append(current_select)
                break
        if i == len(vehicle_range)-1:
            break
    return sampled_frames

def LRE_2_algorithm(result_tuple):
    # 理论上对每辆车辆看到两次需要采样多少帧
    """
    hint: 我们可以把这个问题看作是一个dominant的问题，每一帧采样帧可以覆盖k辆车辆，我们可以把这帧覆盖的
    车辆当成一个已经解决的集合，这样采样新的一帧的时候，我们就可以当作一个全新的集合迭代式地解决最优采样
    的问题。这样针对每辆车至少采样两帧的情况就可以在采样一帧的基础上进行解决，转化成在已经采样一帧的图像上进行计算
    而且采样问题存在着很严重的木桶效应，即短时间出现的车辆限制了采样效率，因为需要保证这些车辆能够至少被看到k次，
    变相相当于限制了采样的范围(采样的范围由行驶时长最小的车辆决定)
    
    更一般的，我们可以在认为实际上的采样是在两个interval的最小交集的基础上不断的更新采样，同时区间最大的帧会在保证
    的采样基础上，获得更多的observation
    """
    
    vehicle_range = []
    sampled_frames = []
    dominate_group = {}
    covered_interval_id = []
    
    for key, value in result_tuple.items():
        vehicle_range.append([result_tuple[key][0],result_tuple[key][1]])
    vehicle_range = sorted(vehicle_range, key=lambda x:x[1])
    sampled_frames.append(vehicle_range[0][1])
    current_select = vehicle_range[0][1]

    for index, interval in enumerate(vehicle_range):
        if current_select < interval[0]:
            current_select = interval[1]
            sampled_frames.append(interval[1])
    for sampled_frame in sampled_frames:
        for index, interval in enumerate(vehicle_range):
            if index in covered_interval_id:
                continue
            else:
                # 枚举所有的区间
                if interval[0] <= sampled_frame and interval[1] >= sampled_frame:
                    if sampled_frame not in dominate_group.keys():
                        dominate_group[sampled_frame] = [index]
                    else:
                        dominate_group[sampled_frame].append(index)
                    covered_interval_id.append(index)
                         
    for key in dominate_group.keys():
        contain_flag = True # 新的采样帧能够覆盖所有帧
        intervals = [vehicle_range[i] for i in dominate_group[key]]
        intervals = sorted(intervals, key=lambda x:x[0])
        sampled_frame = intervals[-1][0]
        for interval in intervals:
            if interval[0] > sampled_frame or interval[1] < sampled_frame:
                contain_flag = False
                break
        if contain_flag:
            sampled_frames.append(sampled_frame)
        
    return sampled_frames            
               
    
    
    

def sub_lre_algorithm(result_tuple,skip=120):
    # 次优采样算法的理论抽帧
    vehicle_range = []
    sampled_frames = []
    for key, value in result_tuple.items():
        vehicle_range.append([result_tuple[key][0],result_tuple[key][1]])
    vehicle_range = sorted(vehicle_range, key=lambda x:x[1])
    sampled_frames.extend([vehicle_range[0][0]])
    current_select = vehicle_range[0][0]
    while True:
        current_gap = []
        for i in range(len(vehicle_range)):
            if vehicle_range[i][0] < current_select and vehicle_range[i][1] > current_select:
                current_gap.append(vehicle_range[i][1])
            elif vehicle_range[i][0] > current_select:
                break
        if len(current_gap)==0:
            current_select += skip
        else:
            current_select = max(current_gap)
        sampled_frames.append(current_select)
        if i == len(vehicle_range)-1:
            break
    return sampled_frames
    
def paint_vehicle_track(video_name,video_path,gt_tuple,tracks):
    
    f = open("./outputs/parsed_results/{}_results.json".format(video_name),"r")
    results_tuple = json.load(f)
    f.close()
    f = open("./outputs/parsed_results/{}_results_dict.json".format(video_name),"r")
    match_dict = json.load(f)
    f.close()
    videoCapture = cv2.VideoCapture(video_path)
    for key, item in results_tuple.items():
        track_id = item[1]
        gt_id = match_dict[key]
        gt_info = gt_tuple[gt_id]
        start,end,start_bbox,end_bbox,_ = gt_info
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, start)
        success, current_image = videoCapture.read() 
        cv2.rectangle(current_image, (start_bbox[0], start_bbox[1]), (start_bbox[2], start_bbox[3]), (0, 255, 0), 2)
        cv2.rectangle(current_image, (end_bbox[0], end_bbox[1]), (end_bbox[2], end_bbox[3]), (0, 255, 0), 2)
        
        for point in tracks[track_id]:
            cv2.circle(current_image,(int(0.5*(point[0]+point[2])),int(0.5*(point[1]+point[3]))),1,(0,0,255),2)
    
        cv2.imwrite("./outputs/reid/%s/%s.jpg"%(video_name,key),current_image)

def analyse_dataset(tuple_dict,cfg=""):
    """
    分析数据集不同类型车辆的数量和对应的平均持续时间
    """
    car_count = 0
    c_d = []
    truck_count = 0
    t_d = []
    bus_count = 0
    b_d = []
    for key, item in tuple_dict.items():
        if item[-1] == 2:
            car_count += 1
            c_d.append(int(item[1]-item[0]))
        elif item[-1] == 7:
            truck_count += 1
            t_d.append(int(item[1]-item[0]))
        else:
            bus_count += 1
            b_d.append(int(item[1]-item[0]))
    object_covered = np.array([0 for i in range(cfg["end_frame"]+1)])
    for key, item in tuple_dict.items():
        for i in range(item[0],item[1]):
            object_covered[min(i,cfg["end_frame"])] = 1
    frame_covered_ratio = 1-(np.sum(object_covered)/cfg["end_frame"])
    return [car_count, truck_count, bus_count, \
        np.mean(np.array(c_d)), np.mean(np.array(t_d)), np.mean(np.array(b_d)),frame_covered_ratio]
    

if __name__ == '__main__':
    
    # extract_frame_from_video('/home/xyc/video_data/Collected_live_videos/standard_split/Taipei_Hires/concat/test/taipei_test.mp4', 500, 'taipei')
    # extract_frame_from_video('/home/xyc/video_data/Collected_live_videos/standard_split/Flat_Creek_Inn/concat/test/flat_test.mp4', 500, 'flat')
    # extract_frame_from_video('/home/xyc/video_data/Collected_live_videos/standard_split/Square_Northeast/concat/test/square_test.mp4', 500, 'square')
    # extract_frame_from_video('/home/xyc/video_data/Collected_live_videos/standard_split/Adventure_Rentals/concat/test/adventure_test.mp4', 500, 'adventure')
    # extract_frame_from_video('/home/xyc/video_data/Collected_live_videos/standard_split/Jackson_Town/concat/test/jackson_test.mp4', 500, 'jackson')

    video_name = "jackson"
    print(video_name)
    label_parsed, result_tuple = get_blazeit_labels(video_name)
    sampled_frames = sub_lre_algorithm(result_tuple,120)
    print("sub-optimal")
    print(len(sampled_frames))
    sampled_frames = LRE_algorithm(result_tuple)
    print("optimal")
    print(len(sampled_frames))
    
    video_name = "taipei"
    print(video_name)
    label_parsed, result_tuple = get_blazeit_labels(video_name)
    sampled_frames = sub_lre_algorithm(result_tuple,120)
    print("sub-optimal")
    print(len(sampled_frames))
    sampled_frames = LRE_algorithm(result_tuple)
    print("optimal")
    print(len(sampled_frames))
    
    video_name = "adventure"
    print(video_name)
    label_parsed, result_tuple = get_blazeit_labels(video_name)
    sampled_frames = sub_lre_algorithm(result_tuple,90)
    print("sub-optimal")
    print(len(sampled_frames))
    sampled_frames = LRE_algorithm(result_tuple)
    print("optimal")
    print(len(sampled_frames))
    
    video_name = "flat"
    print(video_name)
    label_parsed, result_tuple = get_blazeit_labels(video_name)
    sampled_frames = sub_lre_algorithm(result_tuple,300)
    print("sub-optimal")
    print(len(sampled_frames))
    sampled_frames = LRE_algorithm(result_tuple)
    print("optimal")
    print(len(sampled_frames))
    
    video_name = "square"
    print(video_name)
    label_parsed, result_tuple = get_blazeit_labels(video_name)
    sampled_frames = sub_lre_algorithm(result_tuple,300)
    print("sub-optimal")
    print(len(sampled_frames))
    sampled_frames = LRE_algorithm(result_tuple)
    print("optimal")
    print(len(sampled_frames))
    paint_tracks_with_id("adventure")
    
    pass
