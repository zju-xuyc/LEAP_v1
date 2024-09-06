from settings import settings
from settings.settings import video_details
import cv2
import os
from match_object import match_cars_main_updated
from tools.utils import match_last_car_with_gt,paint_image,convert_float
from tools.frame_difference import frame_difference_score
from reid_extractor import feature_extractor
import numpy as np
from tools.utils import calculate_IOU,calculate_intersection
import time

def is_point_in_rectangle(point, rectangle):
    """
    Judge whether input point is in the polygon
    
    Input Params:
    point -- tuple (x, y) 
    rectangle -- quadruple (left, top, right, bottom)
    
    Return:
    True/False
    """
    x, y = point
    left, top, right, bottom = rectangle
    return left <= x <= right and top <= y <= bottom

def allocate_tracks(tracks,stop_area):
    
    """_summary_
        将轨迹按照起始点和终止点聚类，相当于是对车辆的轨迹类型进行一个显式的分类
    Args:
        tracks (_list_): Input of clustered tracks
        stop_area (_list_): 目标进入/离开场景的区域
    """
    
    str2list = {}
    allocated_id = {} # 将traj按照起始点和终止点聚类
    count = 0
    for track in tracks:
        traj_pair = []
        start_point = ((track[0][0]+track[0][2])/2,(track[0][1]+track[0][3])/2)
        end_point = ((track[-1][0]+track[-1][2])/2,(track[-1][1]+track[-1][3])/2)
        
        for i in range(len(stop_area)):
            flag = is_point_in_rectangle(start_point,stop_area[i])
            if flag:
                traj_pair.append(i)
                break
            
        for i in range(len(stop_area)):
            flag = is_point_in_rectangle(end_point,stop_area[i])
            if flag:
                traj_pair.append(i)
                break
            
        if len(traj_pair) < 2:
            # 未匹配到正常轨迹区域
            traj_pair = [-1,-1]
            # continue
        if str(traj_pair) not in str2list.keys():
            str2list[str(traj_pair)] = [traj_pair]
        if str(traj_pair) not in allocated_id.keys():
            allocated_id[str(traj_pair)] = [count]
        else:
            allocated_id[str(traj_pair)].append(count)
            
        count += 1

    video_details.allocate_id = allocated_id
    video_details.traj_type_dict = str2list           
        

def filter_by_status(detects):
    # 动态更新静止区域所需组件
    stop_cars = video_details.stop_cars
    if len(stop_cars.keys()) == 0:
        return detects
    return_detects = []
    for detect in detects:
        match_flag = False
        for bbox in stop_cars.keys():
            if calculate_IOU(list(bbox),detect[:4])>settings.stop_iou_thresh:
                video_details.stop_cars[bbox] += 2
                if stop_cars[bbox]>1:  # 本轮计算的结果不算在内
                    match_flag = True
                    break
        if not match_flag:
            return_detects.append(detect)
            
    for key in video_details.stop_cars.keys():
        video_details.stop_cars[key]-=1 # 没有任何匹配的需要-1，防止误伤因为红绿灯停车的车辆
        
    return return_detects 

def filter_related_detections(detects):
    # 用于和上一帧一起检测是否有相同位置的车辆，更新stop_car_region
    history_car_detection = video_details.last_frame_detections
    if len(history_car_detection)==0: 
        # 没有历史检测需要匹配，直接跳过     
        return detects
    for detect in detects:
        for h_detect in history_car_detection:
            match_stop_region = False
            if calculate_IOU(detect[:4],h_detect[:4])>settings.stop_iou_thresh:
                for stop_region in video_details.stop_cars.keys():
                    if calculate_IOU(list(stop_region),h_detect[:4])>settings.stop_iou_thresh:
                        match_stop_region = True
                        break
                if not match_stop_region:
                    # 加入新的
                    video_details.stop_cars[tuple(h_detect[:4])] = 1
                break
    return detects

def filter_by_predefined_area(detects,cfg):
    
    return_detects = []
    filter_area = cfg["ignore_region"]
    if len(filter_area) == 0:
        return detects
    for detect in detects:
        iou_flag = False
        for bbox in filter_area:
            if calculate_intersection(detect,bbox) > 0.65:
                iou_flag = True
                break
        if not iou_flag:
            return_detects.append(detect)
            
    return return_detects
                
def save_sampled_pics(frame_sampled,frame_list,video_image_path,video_image_path_list):
    
    for frame in frame_sampled:
        img = cv2.imread(os.path.join(video_image_path,video_image_path_list[frame]))
        for record in frame_list[frame]:
            cv2.rectangle(img,(int(record[1]),int(record[2])),(int(record[3]),int(record[4])),(0,255,0),2)
            cv2.putText(img,str(record[0]),(int(record[1]),int(record[2])),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.imwrite('%s.jpg'%(str(frame)),img)

def get_final_tuple_from_detector(tracks,reid_weight,video_path,detect_object,cfg,args,logger):
    
    from tools.analyse_utils import LRE_algorithm, sub_lre_algorithm, LRE_2_algorithm
    # 初始化采样间隔时间
    video_details.skip_frames = cfg["skip_frames"]
    video_details.adaptive_skip = cfg["skip_frames"]

    if args.fixed_sample:

        lre_sampled_frames = LRE_algorithm(video_details.gt_tuple)
        print(len(lre_sampled_frames))
        lre_sampled_frames_2 = LRE_2_algorithm(video_details.gt_tuple)
        print(len(lre_sampled_frames_2))
        exit()
        sub_lre_frames = sub_lre_algorithm(video_details.gt_tuple)
        uniform = [i*128 for i in range(843)]
        sampled_frames = uniform
    
    if len(tracks) == 2:
        tracks = tracks[0]
    
    stop_region = cfg["stop_area"]
    # allocate_id中出现-1的情况，说明没有匹配到正常轨迹区域
    allocate_tracks(tracks, stop_region)
    # 图像特征提取
    extractor = feature_extractor(reid_weight,cfg["w"],cfg["h"])
    extractor.init_extractor()
    # 完整得到所有的车辆 tuple的预测信息
    videoCapture = cv2.VideoCapture(video_path)
    
    if args.fixed_sample:
        current_frame = sampled_frames[0]
        id_sample = 0
        
    else:
        current_frame = cfg["start_frame"]
        
    skip_frames = video_details.skip_frames

    if args.use_mask:
        logger.info("Masking Mode")
        mask_image = cv2.imread("./fixed_files/masks/"+cfg["video_name"]+"_mask.jpg"\
            ,cv2.IMREAD_GRAYSCALE)
        
    # Start time counting
    video_details.start_time = time.time()
    while current_frame < cfg["end_frame"]:

        if args.active_log:
            logger.info("当前采样帧为: %d"%(current_frame))
        decode_time = time.time()
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, current_image = videoCapture.read() 
        original_image = current_image.copy()
        video_details.background_img = current_image
        """
        这样可以进行进一步的加速，缺点是会牺牲一部分的硬盘空间
        """
        # current_image = cv2.imread("/mnt/data_ssd1/xyc/video_data/%s/%d.jpg"%(cfg["video_name"],current_frame))
        # original_image = current_image.copy()
        # video_details.background_img = current_image
        video_details.decode_time += time.time() - decode_time
        if args.use_mask:
            current_image = cv2.add(current_image,np.zeros(np.shape(current_image),dtype=np.uint8),mask=mask_image)  
        
        # if not success:
        if current_image is None:
            print("采样帧越界,程序结束")
            print("当前采样帧为 %d"%current_frame)
            exit()     

        # 在检测之前需要过一下帧差过滤器
        if args.use_filter:
            if video_details.blank_frame:
                # 统计时间
                differ_time = time.time()
                difference_score = frame_difference_score(current_image,video_details.history_frame,cfg["differ_abs_thresh"])
                video_details.frame_differencer_time += time.time() - differ_time

                if args.active_log:
                    logger.info("当前帧与上一帧空帧的相似度为: %5f"%difference_score)
                    
                if difference_score < cfg["difference_thresh"]:
                    current_frame += skip_frames
                    if args.active_log:
                        logger.info("帧差过滤器过滤了一帧")
                    video_details.differencor += 1 # 过滤计数器+1
                    continue

        detect_time = time.time()                
        if args.use_distill:                 
            _, results = detect_object.detect(current_image)
        else:
            results,flag = detect_object(current_image)
            if len(results)>0:
                results = results.cpu().numpy()
        video_details.detector_time += time.time()-detect_time   

        # results{x1,y1,x2,y2,class_id}
        # 获取当前帧的图片
        image_selected = current_image 
        
        # 需要过滤一些对检测可能有影响的区域
        results = filter_by_predefined_area(results, cfg)
        
        # 测试检测结果是否准确，画面有可能一辆车也没有，需要跳过
        if len(results)==0:
            if args.visualize:
                cv2.imwrite("./outputs/selected_frames/%d.jpg"%current_frame,image_selected)
            
            video_details.history_frame = current_image
            video_details.frame_sampled.append(current_frame)
            video_details.blank_frame = True 
            current_frame += skip_frames            
            continue
        
        # 需要确认是否需要采样
        video_details.frame_sampled.append(current_frame)
        video_details.blank_frame = False # 本帧内有物体出现
        # [x1,y1,x2,y2,conf,class]
        results = convert_float(results)

        # 加上过滤的静止车辆，后面可以通过输入的参数控制            
        # if True:
        #     results_threshed = filter_by_status(results_threshed)
        
        # if True:
        #     results_threshed = filter_related_detections(results_threshed)    
        """
        绘制该帧中GT检测框和实际检测框
        """


        if args.visualize:
            cv2.imwrite("./outputs/selected_frames_origin/%d.jpg"%current_frame, original_image)
            image_selected = paint_image(image_selected,video_details.gt_labels[current_frame],results)
            cv2.imwrite("./outputs/selected_frames/%d.jpg"%current_frame, image_selected)

        # 得到需要修正的系数和车辆是否和之前的车辆匹配上
        # 最后出现的车被排在第一个
        # result 要更新
        match_time = time.time()
        sample_gap = match_cars_main_updated(current_frame,current_image,\
            results,extractor,tracks,cfg,args,logger)
        video_details.match_time += time.time()-match_time
        # logger.info("本帧内预测的所有间隔")
        # logger.info(sample_gap)
        # TODO: 找到最合适的采样间隔
        # 目前使用的是当前集合内的最大值
        sample_gap_selected = max(sample_gap)
        
        if sample_gap_selected < cfg["fps"] * cfg["min_gap_time"]: 
            current_frame = current_frame + cfg["fps"] * cfg["min_gap_time"]
            
        elif sample_gap_selected > cfg["fps"] * cfg["max_gap_time"]:

            if args.adaptive_sample:
                # 选择倒数第二个不是最大值的值，增加一定采样帧的数量降低误差
                gap_max = cfg["fps"] * cfg["min_gap_time"]
                for gap in sample_gap:
                    if gap < cfg["fps"] * cfg["max_gap_time"] and gap > gap_max:
                        gap_max = gap
                current_frame += gap_max
            else:
                current_frame += cfg["fps"] * cfg["max_gap_time"]
            
        else:           
            current_frame = sample_gap_selected + current_frame
        
        if args.fixed_sample:
            id_sample+=1
            if id_sample == len(sampled_frames):
                break
            current_frame = sampled_frames[id_sample]
        
        if args.active_log:
            logger.info("选择的采样帧间隔: %d"%(current_frame-video_details.frame_sampled[-1]))
        
        # del_keys = []
        # for key in video_details.stop_cars.keys():
        #     if video_details.stop_cars[key]<0:
        #         del_keys.append(key)
        # for key in del_keys:
        #     del video_details.stop_cars[key]
        # print("当前忽略的区域")
        # print(video_details.stop_cars.keys())
        # video_details.background_img = current_image

        video_details.history_frame = current_image

    return video_details.frame_sampled, video_details.resolved_tuple