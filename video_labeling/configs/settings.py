COCO_CLASSES = [2,5,7]
COCO_OBJECTS = ["car","bus","truck"]

USE_TRAJ_INFO = True  # 使用轨迹信息代替矩阵的信息
ThreshScore = 0.55

coco_names_invert = {7:"truck",2:"car",5:"bus"}
coco_names = {"car":2,"bus":5,"truck":7,"others":0}

IOU_thresh = 0.6
Conf_thresh = 0.45    # 目标检测器的置信度阈值
stop_iou_thresh = 0.7 # 判断是否是同一辆停下的车

seed = 100            # 固定随机种子
K = 6400              # 采样帧的数量
end_frame = 106500 # 106500
traj_thresh_time = 2  # 最短轨迹需要持续的时间


label_base = {
                "detrac": "/home/xyc/video_data/DETRAC/xml_",
                "blazeit": "/home/xyc/video_data/pseudo_labels",
                "m30": "/home/xyc/video_data/GRAM-RTMv4/Annotations/",
                "customized":"/home/xyc/video_data/pseudo_labels",
            }

image_base = {
                "detrac":"/home/xyc/video_data/DETRAC/images",
                "m30": "/home/xyc/video_data/",
                "customized":""
            }

video_resol_patch = { # resol fps patch_width
                "detrac":[[960,540],25,30],
                "jackson":[[1920,1080],30,60],
                "taipei":[[1280,720],30,40],
                "m30":[[800,480],25,40],
                "m30_hd":[[1200,720],25,40],
                "adventure":[[1920,1080],30,60], # 原始视频是30fps 但是重新写入之后调成了25fps 所以按25fps来算(更正，已经恢复正常)
                "flat":[[1920,1080],30,60],
                "gather":[[1920,1080],30,60],
                "square":[[1920,1080],30,60],
                "caldot":[[720,480],15,30],
            }

map_dict = {"adventure":"Adventure_Rentals","flat":"Flat_Creek_Inn",\
                "gather":"Gather","square":"Square_Northest"}

stop_region = { "jackson": [[453,563,753,728],[1345,510,1670,624],[553,783,1665,1079]],
               "taipei":[[],[]]
                }

traffic_light = {"jackson": [[550,1690],[1140,720]]}

class Video_info(object):
    # 解析过程中需要用到的所有参数
    def __init__(self):
        
        self.video_name = "void"
        self.v_width = 0              # 视频帧宽度
        self.v_height = 0             # 帧高度
        self.fps = 0                  # 帧速率
        self.skip_frames = 30         # 当没有出现物体时，需要调跳过的帧数
        self.stop_thresh = 600         # 区分车辆是否有停顿的时间阈值
        self.ThreshLen = 0
        """
        用于确定方向,TODO: 适应多车道的情况
        """        
        self.reverse = True           # 跟画面车辆出现的相对位置有关
        self.direction = "vertical"   # 车辆运行方向 ["vertical","horizon"]

        self.traj_dict = {}           # 存储预处理中解析出的轨迹
        self.traj_match = {}          # 存储汽车id对应的轨迹
        self.gt_labels = []           # 每个视频对应的真实标签
        self.gt_tuple = {}            # 每个视频对应的真实解析输出 {car_id:[start_time,end_time]}
        self.match_dict = {}          # key: 解析中分配的车辆ID value: GT中的车辆ID
        self.object_type = {}         # key: 解析中分配的车辆ID value: object_type
        self.return_tuple = {}        # 返回的结果
        '''
        "vehicle_id": [estimated_starttime,estimated_endtime], tracklet_id, class_id
        '''
        self.stop_cars = {}           # {(x1,y1,x2,y2):hit_num} 存储视频中静止的干扰车辆的位置(可能还需要加上时间)
        # 过程中临时共享变量
        self.history_cache = []       # 存储历史检测的集合 
        '''
            0: vehicle_id, 
            1 - 4: x1, y1, x2, y2, 
            5: tracklet_id, 
            6: car_image,
            7: nearest_tracklet_point_location,
            8: class_id
        '''
        self.frame_sampled = []       # 采样帧ID
        self.stop_area = []
        self.reid_acc = [0,0]
        
        self.visualize = True
        self.load = False
        self.use_mask = True
        
        #* Detector System
        self.detector_name = "yolov5x.engine"
        
        #* Sample System
        self.sample_type = "uniform"
        self.frame_gap = 16
        
        #* Tracker System
        self.track_type = "dyte"
        self.track_thresh = 0.7
        self.sampling_rate = 10
        self.frame_rate = 30
        self.track_buffer = 30
        self.stdp = 1. / 20
        self.stdv = 1. / 80
        self.stda = 1. / 180
        self.adjusted_gate = 1.5
        self.proximity_thresh = 0.65
        self.appearance_thresh = 0.5
        self.match_thresh_d1 = 0.85
        self.match_thresh_d2 = 1.0

        #* Visualizer System
        self.color_dict = {}
    
    def from_args(self, args, skip_args=['load', 'fps', 'visualize', 'reverse', 'direction']):
        import re        
        args_attrs = list(vars(args).keys())
        strongotif_attrs = list(filter(lambda x: not(x[:2] == "__" or x[-2:] == "__"), list(set(self.__dir__()) ^ set(object().__dir__()))))
        canchange_attrs = list(set(args_attrs) & set(strongotif_attrs) - set(skip_args))
        [setattr(self, canchange_attr, getattr(args, canchange_attr)) for canchange_attr in canchange_attrs if getattr(args, canchange_attr) is not None]
        
    def __str__(self):  # 定义打印对象时打印的字符串
        return f'''
                   track_thresh: {self.track_thresh}
                   match_thresh_d1: {self.match_thresh_d1}
                   match_thresh_d2: {self.match_thresh_d2}
                   frame_gap: {self.frame_gap}
                   adjusted_gate: {self.adjusted_gate}'''
        
video_details = Video_info()

# jacksontown红灯时长24s(720帧) 绿灯时长38s(1140帧) 绿灯开始时间550帧