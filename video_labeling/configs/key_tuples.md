### Inference过程中使用的变量

- tracks: 一串轨迹点组成的向量列表，这里的中心点的选择可以使用中心点或者下边缘中点，选用的是1. 过滤后的无停顿正常轨迹 2. 过滤为纯汽车轨迹(轨迹密度大)
- mask: 掩码矩阵，遮挡无关背景，可以降低检测误差
- video_details.frame_sampled: 列表，每次采样时记录采样帧ID
- 目标检测器的结果 detect: [x1,y1,x2,y2,conf,class_id]

match_cars_main


history_detect: [分配_id,x1,y1,x2,y2,car_image, track_id, point_location, object_type]

detections: [x1,y1,x2,y2,conf,class_id]

return_tuple: {"car_id":[[start,end],track_id,class_id]}
