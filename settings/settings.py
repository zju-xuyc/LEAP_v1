### 用于存放永久性设置以及实验过程数据

# FIXME Should put in seperate yaml files

CODE_ROOT = "/home/xuyanchao"
LABEL_BASE = "/home/xuyanchao/video_data/pesudo_labels"
VIDEO_BASE = "/home/xuyanchao/video_data/COllected_live_videos"

IoU_min = 0.3
cluster_min_traj_num = 12
distill_resol = 640

video_path = "%s/video_data/Collected_live_videos"%CODE_ROOT
label_path = {"blazeit":"%s/video_data/pseudo_labels"%CODE_ROOT,"m30":"%s/video_data/M30/annotations"%CODE_ROOT}
mask_path = "%s/video_data/masks/"%CODE_ROOT

# need update when add new files
map_dict = {"adventure":"Adventure_Rentals","flat":"Flat_Creek_Inn",\
                "gather":"Gather","square":"Square_Northeast","jackson":"Jackson_Town",\
                    "taipei":"Taipei_Hires"}

coco_names_invert = {7:"truck",2:"car",5:"bus"}
coco_names = {"car":2,"bus":5,"truck":7,0:"others","van":2,"big-truck":7}





class Video_info(object):
    # 解析过程中需要用到的所有参数
    def __init__(self):
        
        self.background_img = 0
        """
        用于确定方向,TODO: 适应多车道的情况
        """        
        self.reverse = True           # 跟画面车辆出现的相对位置有关
        self.direction = "vertical"   # 车辆运行方向 ["vertical","horizon"]
        self.skip_frames = 50         # 跳帧数
        self.adaptive_skip = 90       # 自适应跳帧数

        self.traj_dict = {}           # 存储预处理中解析出的轨迹
        self.allocate_id = {}         # 存储不同轨迹类型对应的解析好并聚类好轨迹的id 
        self.traj_type_dict = {}      # 存储轨迹类型到起始点位置的映射
        self.gt_labels = []           # 每个视频对应的真实标签
        self.gt_tuple = {}            # 每个视频对应的真实解析输出 {car_id:[start_time,end_time]}
        self.gt_tuple_origin = {}
        self.match_dict = {}          # key: 解析中分配的车辆ID value: GT中的车辆ID
        self.object_type = {}         # key: 解析中分配的车辆ID value: object_type
        self.return_tuple = {}        # 返回的最终结果
        self.stop_cars = {}           # {(x1,y1,x2,y2):hit_num} 存储视频中静止的干扰车辆的位置(可能还需要加上时间)
        
        # 过程中临时共享变量
        self.history_cache = []       # 存储上一帧历史检测的集合
        self.history_frame = 0        # 上一帧的图像
        self.blank_frame = False      # 标志当前的帧是否为空
        self.resolved_tuple = {}      # 存储汽车id对应的轨迹候选集合以及观测的结果
        self.frame_sampled = []       # 采样帧ID
        self.bbox_rate = {}           # 存储每辆车对应的bbox和采样点的位置
        self.stop_area = []

        # 用于统计实验中过程数据
        self.reid_acc = [0,0]
        self.differencor = 0
        self.start_time = 0
        self.detector_time = 0
        self.reid_time = 0
        self.frame_differencer_time = 0
        self.match_time = 0
        self.decode_time = 0
        self.assign_time = 0
        
        self.visualize = True
        self.load = False
        self.use_mask = True

video_details = Video_info()