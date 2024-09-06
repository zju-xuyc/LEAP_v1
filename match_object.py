from tools.utils import calculate_IOU
from tools.utils import match_last_car_with_gt
from settings.settings import video_details
import cv2
import time

def get_point_update(detect,tracks,cfg):
    
    """
    针对最新的算法更新
    """ 
    dis_thresh = cfg["dis_thresh"] # 这个和场景有关，后面要优化(md，又是一个参数)
    # key = 分配的起点和终点 value = 最接近的轨迹id,最接近的点的位置,距离
    match_dict = {}
    allocate_id = video_details.allocate_id
    # 首先选取车辆的中间点
    detect_mid_x = (detect[0]+detect[2])/2
    detect_mid_y = (detect[1]+detect[3])/2
    match_flag = False
    # 根据车辆的起始位置进行轨迹的分配
    """
    每个类型的轨迹只需要匹配一个即可
    """
    for key in allocate_id.keys():
        # 这里先只存储每条潜在运行路线中最近的一条
        distance = 1e10
        candidate_traj_id = -1
        nearest_point_location = -1
        for id in allocate_id[key]:
            pos_count = 0            
            for point in tracks[id]:
                x_mid = (point[0]+point[2])/2
                y_mid = (point[1]+point[3])/2
                dis = (detect_mid_x-x_mid)**2 + (detect_mid_y-y_mid)**2
                if dis < distance:
                    distance = dis
                    candidate_traj_id = id
                    nearest_point_location = pos_count
                pos_count += 1
        if distance < dis_thresh and key!="[-1, -1]":
            match_dict[key] = [candidate_traj_id,nearest_point_location,distance]
            match_flag = True            
    
    if not match_flag:
        # 没有匹配上任何的轨迹随机分配一个
        # match_dict["[-1, -1]"] = [allocate_id["[-1, -1]"][0],60,4000]
        match_dict["[66, 66]"] = [5,60,cfg["dis_thresh"]]
    return match_dict

def sort_detections(detctions,direction="vertical",reverse=True):
    # 用于将检测到的目标按照位置先后顺序进行排序，默认最后一辆车是排在最前面
    # detections [x1,y1,x2,y2,...]
    """
    升序 False, 降序 True 默认是升序
    自底向上行驶: vertical+ True
    自顶向下行驶: vertical+ False
    自左向右行驶: horizon + False
    自右向左行驶: horizon + True
    """
    detections_sorted = []
    if direction=="vertical":
        detections_sorted = sorted(detctions,key=lambda x:x[1],reverse=reverse)
    if direction=="horizon":
        detections_sorted = sorted(detctions,key=lambda x:x[0],reverse=reverse)
    return detections_sorted

def match_cars_main_updated(current_frame,curr_frame_img,records,\
    extractor,tracks,cfg,args,logger):
    """
    更新后的车辆匹配,不使用matrix信息
    """
    # 进行车辆的匹配
    reverse = video_details.reverse
    direction = video_details.direction
    # 目前先简单排序(大部分情况下不需要，简单场景下有助于提高精度)
    if cfg["video_name"] in ["m30","m30hd"]:
        records = sort_detections(records, direction, reverse)
    # 判断之前是否已经有历史车辆
    """
    history_detect: [分配_id,x1,y1,x2,y2,track_id, car_image,point_location, object_type]
    records: [x1,y1,x2,y2,conf,class_id]
    return_tuple: {"car_id":[[start,end],track_id,class_id]}
    每次更新: history_detect; return_tuple TODO: tracks;    
    """
    """
    加入时间区分轨迹 
    TODO: 检查是否有必要保留长轨迹,目前来看长时间的轨迹仅有参考价值
    """
    stop_region = cfg["stop_area"]

    sample_gap = []
    reid_count = 0
    reid_count = 0
    
    if current_frame == cfg["start_frame"] or len(video_details.frame_sampled)==1:
        # 没有历史检测的情况，全部是新车，不可能有匹配的车辆
        if args.active_log:
            logger.info("No history detection, all cars are new!")

        for record in records:
            x1, y1, x2, y2, conf, class_id = record
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            car_image = curr_frame_img[y1:y2,x1:x2,:]
            # match_traj_dict key: moving_pattern_id [candidate_traj_id,nearest_point_location,distance]
            start = time.time()
            match_traj_dict = get_point_update(record,tracks,cfg)
            video_details.assign_time += time.time() - start
            # 直接按顺序分配ID
            apply_id = max(video_details.resolved_tuple.keys())+1 \
                if len(video_details.resolved_tuple.keys())!=0 else 1
                
            # 增加到存储上一帧检测信息的车辆以及匹配信息    
            video_details.history_cache.append([apply_id,x1,y1,x2,y2,class_id,match_traj_dict,\
                car_image])
            # [历史检测，匹配轨迹，车辆图像，估计的轨迹时间]
            video_details.resolved_tuple[apply_id]=[[[x1,y1,x2,y2,class_id,current_frame]],match_traj_dict,\
                [car_image],[]]
            # 直接通过匹配到轨迹的位置得到估计的车辆起止时间
            from tools.utils import get_sample_gap
            intervals, closest_interval = get_sample_gap(current_frame,tracks,match_traj_dict)
            video_details.resolved_tuple[apply_id][-1] = [closest_interval]

            for interval in intervals:
                sample_gap.append(interval[1]-current_frame)

            # 匹配车辆 ID:
            id_match = match_last_car_with_gt(record,video_details.gt_labels[current_frame],current_frame,cfg)
            video_details.match_dict[apply_id] = id_match
            video_details.object_type[apply_id] = class_id

        return sample_gap
    
    else:
        # 当前已经有车辆 records
        exclude_list = [] # 只考虑1对1的情况,当车辆已被匹配时，直接忽略 
        current_cache = []
        for record in records:
            x1,y1,x2,y2,conf,class_id = record
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            car_image = curr_frame_img[y1:y2,x1:x2,:]
            # 得到初步匹配的轨迹信息
            start = time.time()
            match_traj_dict = get_point_update(record,tracks,cfg)
            video_details.assign_time += time.time() - start
            selection_gap = int(current_frame-video_details.frame_sampled[-2])
            match_flag = False
            
            for history_obj in video_details.history_cache: 
                bbox_hist = history_obj[1:5]
                match_iou = calculate_IOU([x1,y1,x2,y2],bbox_hist)
                apply_id = history_obj[0] 
                from tools.utils import justify_intersect
                match_intersect = justify_intersect(history_obj,match_traj_dict)
                
                # 首先判断是否和前一帧的物体位置上的匹配(即车辆是否静止)
                if match_iou > cfg["stop_iou_thresh"] and apply_id not in exclude_list:
                    # 增加类型的匹配(目前暂不考虑)
                    # 当超过判断停车的阈值时，自动认定车辆为停车状态(TODO:是否需要额外定义一个空间用于存放被认为是已经静止的车辆)     
                    current_cache.append([apply_id,x1,y1,x2,y2,class_id,match_traj_dict,\
                                            car_image])
                    video_details.resolved_tuple[apply_id][0].append([x1,y1,x2,y2,class_id,current_frame])
                    
                    # 更新return tuple 加上中间等待的间隔
                    interval_new = []
                    for interval in video_details.resolved_tuple[apply_id][-1]:
                        interval_new.append([interval[0],interval[1]+selection_gap])
                        
                    video_details.resolved_tuple[apply_id][-1] = interval_new
                    video_details.resolved_tuple[apply_id][2].append(car_image)
                    
                    exclude_list.append(apply_id) # 该车不会再进入匹配过程
                    match_flag = True
                    # 先和没车时一样，设定为默认跳过的数量
                    sample_gap.append(cfg["skip_frames"])
                    current_cache.append([apply_id,x1,y1,x2,y2,class_id,match_traj_dict,\
                                            car_image])
                    break
                    
                # elif candidate_traj_id == history_obj[5] and class_id==history_obj[-1]:
                elif (match_intersect) and (apply_id not in exclude_list):
                    # match_intersect [key:[[hist_traj,match_time,distance],[now_traj,match_time,distance]]]
                    # 和上一帧的车辆隶属于同一条轨迹
                    # match_traj_history = history_obj[-2]
                    last_frame_id = video_details.frame_sampled[-2]
                    # [track_id, nearest_point_location, distance]
                    traj_hist = match_intersect[1][0]
                    traj_curr = match_intersect[1][1]
                    # 计算两者预测的interval
                    if traj_hist[0] == traj_curr[0] and traj_hist[1] > traj_curr[1]:
                        # 匹配的方向错误，反方向轨迹
                        hist_interval = [last_frame_id-(len(tracks[traj_hist[0]]) - traj_hist[1]),last_frame_id + \
                                        traj_hist[1]]
                        curr_interval = [current_frame-(len(tracks[traj_hist[0]]) - traj_hist[1]),current_frame + \
                                        traj_hist[1]]
                    else:
                        hist_interval = [last_frame_id-traj_hist[1],last_frame_id + \
                                        (len(tracks[traj_hist[0]]) - traj_hist[1])]
                        curr_interval = [current_frame-traj_curr[1],current_frame + \
                                        (len(tracks[traj_curr[0]]) - traj_curr[1])]
                        
                    from tools.utils import justify_intersect_rate
                    MAPE_GAP = justify_intersect_rate(hist_interval,curr_interval)
                    if MAPE_GAP > cfg["intersect_overlap_thresh"]: # 误差相差不大，可以认为是速度上的差异
                        
                        apply_id = history_obj[0]                                                        
                        current_cache.append([apply_id,x1,y1,x2,y2,class_id,match_traj_dict,\
                                    car_image])
                        video_details.resolved_tuple[apply_id][0].append([x1,y1,x2,y2,class_id,current_frame])
                        # 可以唯一确定轨迹了
                        video_details.resolved_tuple[apply_id][1] = {match_intersect[0]:match_traj_dict[match_intersect[0]]}
                        video_details.resolved_tuple[apply_id][2].append(car_image)
                        exclude_list.append(apply_id)
                        match_flag = True
                        sample_gap.append(min(hist_interval[1],curr_interval[1])-current_frame)
                        # 进一步矫正interval
                        if traj_hist[0] == traj_curr[0]:
                            error_rate = abs(traj_hist[1]-traj_curr[1])/selection_gap
                            curr_interval = [min(hist_interval[0],curr_interval[0]),max(hist_interval[1],curr_interval[1])]
                        video_details.resolved_tuple[apply_id][-1] = [curr_interval]
                        break
                    
                    else:
                        # 时间上不支持，加入reid的部分进一步测试        
                        curr_frame = video_details.frame_sampled[-1]
                        last_frame = video_details.frame_sampled[-2]
                        if cfg["dataset_group"] == "blazeit":
                            curr_bbox = [x1,y1,x2,y2]
                            last_bbox = history_obj[1:5]
                            history_img = history_obj[-1]

                            reid_time = time.time()
                            image_score = float(extractor.inference_pic([car_image,history_img,\
                                curr_bbox,last_bbox,curr_frame,last_frame]).item())
                            video_details.reid_time += time.time()-reid_time
                            
                        else:
                            image_score = 0.1 # 默认都不匹配

                        if args.visualize:
                            reid_count+=1
                            cv2.imwrite("./outputs/reid/%s/%d_%d_%s.jpg"%(cfg["video_name"],current_frame,reid_count,image_score),car_image)
                            reid_count+=1
                            cv2.imwrite("./outputs/reid/%s/%d_%d.jpg"%(cfg["video_name"],current_frame,reid_count),history_img)  
                                
                        if args.visualize:
                            """测试Reid是否准确"""
                            curr_label = video_details.gt_labels[curr_frame]
                            last_label = video_details.gt_labels[last_frame]
                            curr_id = -1
                            last_id = -2
                            for label in curr_label:
                                thresh = calculate_IOU(curr_bbox,label[1:5])
                                if thresh>0.8:
                                    curr_id = label[0]
                                    break
                            for label in last_label:
                                thresh = calculate_IOU(last_bbox,label[1:5])
                                if thresh>0.8:
                                    last_id = label[0]
                                    break
                            if curr_id==last_id and image_score > cfg["image_thresh_score"]:
                                video_details.reid_acc[0]+=1
                            elif curr_id!=last_id and image_score < cfg["image_thresh_score"]:
                                video_details.reid_acc[0]+=1
                            else:
                                # 匹配错误
                                video_details.reid_acc[1]+=1                            
                        # logger.info("车辆相似度")
                        # logger.info(image_score)
                            
                        if image_score > cfg["image_thresh_score"]:
                            
                            # 同一辆车
                            apply_id = history_obj[0]
                            current_cache.append([apply_id,x1,y1,x2,y2,class_id,match_traj_dict,\
                                        car_image])
                            video_details.resolved_tuple[apply_id][0].append([x1,y1,x2,y2,class_id,current_frame])
                            # 连续两次匹配，可以确定唯一轨迹了
                            video_details.resolved_tuple[apply_id][1] = {match_intersect[0]:match_traj_dict[match_intersect[0]]}
                            video_details.resolved_tuple[apply_id][2].append(car_image)
                            # 更新return tuple(目前先不处理)
                            exclude_list.append(apply_id)
                            match_flag = True
                            traj_curr = match_traj_dict[match_intersect[0]]
                            curr_interval = [current_frame-traj_curr[1],current_frame + \
                                     (len(tracks[traj_curr[0]]) - traj_curr[1])]
                            sample_gap.append(curr_interval[1] - current_frame)
                            break
                        else:
                            pass
                        
            if not match_flag:
                # 没有匹配上任何历史车辆，可以认为是新车了
                apply_id = max(video_details.resolved_tuple.keys())+1 \
                    if len(video_details.resolved_tuple.keys())!=0 else 1
                    
                current_cache.append([apply_id,x1,y1,x2,y2,class_id,match_traj_dict,\
                                            car_image])
                video_details.resolved_tuple[apply_id]=[[[x1,y1,x2,y2,class_id,current_frame]],match_traj_dict,\
                                            [car_image],[]]
                
                # 直接通过匹配到轨迹的位置得到估计的车辆起止时间
                from tools.utils import get_sample_gap
                intervals, closest_interval = get_sample_gap(current_frame,tracks,match_traj_dict)
                video_details.resolved_tuple[apply_id][-1] = [closest_interval]
                
                for interval in intervals:
                    sample_gap.append(interval[1]-current_frame)
                
                id_match = match_last_car_with_gt(record,video_details.gt_labels[current_frame],current_frame,cfg)
                
                video_details.match_dict[apply_id] = id_match
                video_details.object_type[apply_id] = class_id
    
    video_details.history_cache = current_cache
    return sample_gap

