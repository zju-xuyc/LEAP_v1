import os
import shutil
import multiprocessing
from traj_cluster.distance import distance
import numpy as np
import math
from traj_cluster.cluster.Kmediods import Kmediods
from evaluate import DBI_Vec
from tools.analyse_utils import get_blazeit_labels, get_m30_labels 
from settings import settings

def compute_distance(tr_id,trajs_dict,distance_type="frechet"):
    print("Begin Compute: ",tr_id)
    # if not os.path.exists("./traj_cluster/frechet_distance"):
    #     os.makedirs("./traj_cluster/frechet_distance")
    # 只需要计算大于等物当前id的序列
    try:        
        trs_matrix = distance.cdist(trajs_dict,{tr_id:trajs_dict[tr_id]},type=distance_type)  
    except Exception as e:
        print(e)
    np.save('./outputs/traj_compute_tmp/'+str(tr_id)+".npy",trs_matrix)
    print("Done! ",tr_id)

def compute_distance_single(tr_id,trajs_dict,distance_type="frechet"):
    try:
        trs_matrix = distance.cdist(trajs_dict,{tr_id:trajs_dict[tr_id]},type=distance_type)  
    except Exception as e:
        print(e)
    return trs_matrix
    

def compute_distance_all(traj_dict,num_thread=48):

    shutil.rmtree('./outputs/traj_compute_tmp')
    os.mkdir('./outputs/traj_compute_tmp')
    
    pool = multiprocessing.Pool(processes=num_thread)
    for tr_id in traj_dict.keys():
        # 可能存在进程锁的问题
        pool.apply_async(compute_distance, (tr_id,traj_dict))
    pool.close()
    pool.join()
    
def compute_distance_all_single(traj_dict):
    distances = {}
    for tr_id in traj_dict.keys():
        # 可能存在进程锁的问题
        trs_matrix = compute_distance_single(tr_id,traj_dict)
        for id in trs_matrix:
            distances[id] = trs_matrix[id]
    return distances

def load_computed_distance():

    files = os.listdir("./outputs/traj_compute_tmp/")
    
    distances = {}
    for file in files:
        distance_dict = np.load(os.path.join("./outputs/traj_compute_tmp",file),allow_pickle=True)
        distance_dict = distance_dict.item()
        for id in distance_dict:
            distances[id] = distance_dict[id]
            id2 = tuple((id[1],id[0]))
            if id2 not in distances.keys():
                distances[id2] = distance_dict[id]

    return distances

def convert2dict(tracks,split_ratio=2):
    # 加入采样系数
    traj_dict = {}
    count = 0
    for track in tracks:
        traj_dict[count] = []
        point_count = 0
        for detect in track:
            if point_count%split_ratio==0:
                traj_dict[count].append([int(0.5*detect[0]+0.5*detect[2]),int(0.5*detect[1]+0.5*detect[3])])
            point_count+=1
        count += 1
    return traj_dict

def convert2matrix(distance_dict):
    numbers = int(math.sqrt(len(distance_dict.keys())))
    distance_matrix = np.zeros((numbers,numbers))
    for key in distance_dict.keys():
        distance_matrix[key[0]][key[1]] = distance_dict[key]
    return distance_matrix

def get_traj_cluster(tracks,k_min=8,k_max=32):

    tracks_origin = tracks.copy()
    traj_dict = convert2dict(tracks)
    compute_distance_all(traj_dict)
    distances = load_computed_distance()
    # distances = compute_distance_all_single(traj_dict)
    distance_matrix = convert2matrix(distances)
    score_min = 1000
    center_result = 0
    for k_num in range(k_min,k_max,2):
        labels, centers = Kmediods(k_num, distance_matrix)

        cluster_dict = {}
        for num in range(len(labels)):
            if labels[num] in cluster_dict.keys():
                cluster_dict[labels[num]].append(num)
            else:
                cluster_dict[labels[num]]=[num]

        selected_centers = []
        for center in centers:
            selected_centers.append(tracks_origin[center])

        score = DBI_Vec(cluster_dict,distance_matrix)

        if score < score_min:

            score_min = score
            center_result = selected_centers

    return center_result, score_min

# Get Trajectory from local files
import cv2

class cluster_traj(object):

    def __init__(self,video_name,background_img,origin_tuples,min_traj_num,thresh_num=100):

        self.video_name = video_name
        self.track_filtered = []
        self.origin_tracks = []
        self.track_clustered = []
        self.background = background_img
        self.origin_tuples = origin_tuples
        self.min_traj_num = min_traj_num
        self.thresh_num = thresh_num # 控制轨迹总数

    def filter_traj(self):

        tracks = []
        for key,value in self.origin_tuples.items():
            tracks.append(value[2])
        self.origin_tracks = tracks

        min_len = 20
        max_len = 600
        min_dis = 100

        for track in tracks:

            if len(track) < max_len and len(track) > min_len:

                if math.sqrt((track[0][0]-track[-1][0])*(track[0][0]-track[-1][0]) + \
                    (track[0][1]-track[-1][1])*(track[0][1]-track[-1][1]))>min_dis:                
                    self.track_filtered.append(track)

        self.track_filtered = self.track_filtered[:self.thresh_num]

    def get_cluster(self):
        tracks_clustered, _ = get_traj_cluster(self.track_filtered,min(\
            self.min_traj_num,len(self.track_filtered)),\
                int(0.5*len(self.track_filtered)))
        self.track_clustered = tracks_clustered
        print(len(tracks_clustered))
        if True:
            colors = [(255,255,255),(0,0,0),(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,100,255),(0,255,255)]
            img_origin = self.background.copy()
            count = 0
            for track in self.origin_tracks:
                count += 1
                for point in track:
                    cv2.circle(img_origin,(int(0.5*(point[0]+point[2])),int(0.5*(point[1]+point[3]))),1,colors[count%8],2)
            cv2.imwrite("./%s_trajs_origin.jpg"%(self.video_name),img_origin)

            img_clustered = self.background.copy()
            count = 0
            for track in self.track_clustered:
                count += 1
                for point in track:
                    cv2.circle(img_clustered,(int(0.5*(point[0]+point[2])),int(0.5*(point[1]+point[3]))),1,colors[count%8],2)
        
            cv2.imwrite("./%s_trajs_clustered_s.jpg"%(self.video_name),img_clustered)

def track_cluster_from_labels(video_name,v_type="blazeit",cluster_min_num=16,cluster_tracks_num=96,k=2000,save_result=True):
    """
    """
    if v_type == "blazeit":
        image_background = cv2.imread("./fixed_files/masks/background/%s.jpg"%(video_name))
        label_parsed, tuple_dict, label_tuple_origin = get_blazeit_labels(video_name)
        sorted_keys = sorted(tuple_dict, key=tuple_dict.get)[:k]
        result_by_key = {key: tuple_dict[key] for key in sorted_keys}
        tuple_dict = result_by_key

    else:
        image_background = cv2.imread(settings.VIDEO_BASE,"/M30/%s_mask/image000001.jpg"%(video_name))
        label_parsed, tuple_dict = get_m30_labels(video_name,k)

    cluster = cluster_traj(video_name,image_background,tuple_dict,cluster_min_num,cluster_tracks_num)
    cluster.filter_traj() 
    cluster.get_cluster()

    if save_result:
        np.save("./fixed_files/preprocessed/%s/"%(video_name)+video_name+"_0_%d_tracks_clustered.npy"%(k),cluster.track_clustered)
        if v_type == "blazeit": 
            np.save("./fixed_files/preprocessed/%s/"%(video_name)+video_name+"_0_%d_tracks_filtered.npy"%(k),cluster.track_filtered)
        np.save("./fixed_files/preprocessed/%s/"%(video_name)+video_name+"_0_%d_tracks_origin.npy"%(k),cluster.origin_tracks)

if __name__ == "__main__":
    pass