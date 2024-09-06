from leap_utils import * #FIXME Bad import habit
import argparse
from loguru import logger
import time
import datetime
from tools.data_prepare import * #FIXME Bad import habit
from settings.settings import video_details
from detector import Detector, Detector_origin
from configs import cfg
from inference import get_final_tuple_from_detector
from evaluate import parse_intervals,evaluate_query_result,evaluated_object_recall
from leap_utils import paint_traj,evaluate_traj,evaluate_traj_acc
import json

def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/jackson.yaml")
    parser.add_argument("--weight", type=str, default="./weights/best.pt", help="Weight path for distill model") 
    parser.add_argument("-rw","--reid_weight",default=\
                            "/home/xuyanchao/video_data/reid_models/logs")
    
    parser.add_argument("--use_cluster",action="store_false")    # Clustering to get better pattern
    parser.add_argument("--use_distill",action="store_false")    # Using distilled model instead of standard YOLO   
    parser.add_argument("--use_filter",action="store_false")     # Using filter to skip similar frames
    parser.add_argument("--use_hand_label",action="store_false") # Using hand annotated labels instead of pesudo labels
    parser.add_argument("--fixed_sample",action='store_true')    # Using fixed intervals
    parser.add_argument("--type",type=str,default="test", help=["train","test","val"])        
    parser.add_argument("--adaptive_sample",action="store_false")
    parser.add_argument("--use_external_pattern",action="store_true")
    parser.add_argument("--cross_roads",action="store_true")     

    parser.add_argument("-vis","--visualize",action='store_true')
    parser.add_argument("--active_log",action='store_true')
    parser.add_argument("-sr","--save_result",action="store_false",help="Save parsed result")
    parser.add_argument("-ld","--load",action='store_false',help="load file from dict")
    parser.add_argument("--use_label",action='store_true',help="Use preprocesed label instead of bytetracker")
    parser.add_argument("--use_mask",action='store_false',help="Use mask while parsing")
    parser.add_argument("--use_cache",type=str,default="/home/xuyanchao/video_data/convert_ssd") # Using Cache to accelerate
    parser.add_argument("--device",type=str,default='1')

    return parser

def run_leap(cfgs,args,logger):

    dataset_object = get_label_details(cfgs,args,logger)
    label_parsed, tuple_dict, label_tuple_origin = dataset_object.get_label()

    logger.info("%s %s label is parsed!"%(cfgs["video_name"],args.type))

    video_details.gt_labels = label_parsed
    video_details.gt_tuple = tuple_dict
    video_details.gt_tuple_origin = label_tuple_origin

    # First time use, executing clustering process
    if not args.load and cfgs["dataset_group"]=='blazeit':  # Defualt use 'blazeit' as the type of new dataset
        
        from initialize import initialize
        tracks, preprocess_tuple = initialize(cfgs,args,logger)
        
    else:
        try:
            # Use first several minutes of testset to construct pattern
            if not args.use_external_pattern:

                if args.use_cluster:
                    tracks = np.load("./fixed_files/preprocessed/%s/"%(cfgs["video_name"])+cfgs["video_name"]+\
                        "_0_%d_tracks_clustered.npy"%(cfgs["start_frame"]),allow_pickle=True)
                else:
                    tracks = np.load("./fixed_files/preprocessed/%s/"%(cfgs["video_name"])+cfgs["video_name"]+\
                        "_0_%d_tracks_origin.npy"%(cfgs["start_frame"]),allow_pickle=True)
                
                if len(tracks)>2:
                    # FIXME 历史遗留问题，需要检查一下数据
                    tracks_new = []
                    for track in tracks:
                        if len(track)<cfgs['max_len']*1.5:
                            tracks_new.append(track)
                    tracks = [tracks_new,[]]
                else:
                    tracks_new = []
                    for track in tracks[0]:
                        if len(track)<cfgs['max_len']*1.5:
                            tracks_new.append(track)
                    tracks = [tracks_new,[]]

            # Use trajectories in train set to initialize pattern
            else:
                tracks = np.load("./fixed_files/preprocessed/%s/"%(cfgs["video_name"])+cfgs["video_name"]+\
                    "_0_108000_tracks_clustered.npy",allow_pickle=True)
                tracks_new = []
                for track in tracks:
                    if len(track)<cfgs['max_len']: # filter more trajectories
                        tracks_new.append(track)
                tracks = [tracks_new,[]]

        except Exception:
            logger.info("Wrong Input Path:"+"./fixed_files/preprocessed/"+\
                cfgs["video_name"]+"_0_%d_tracks_clustered.npy"%(cfgs["start_frame"]))
            exit()

    # Load object detector
    if args.use_distill:
        logger.info("Using Distilled Model")
        detector = Detector(args.weight,settings.distill_resol,args.device,\
                            iou_thres = cfgs['iou_thresh'],conf_thres = cfgs['conf_thresh'])
    else:
        logger.info("Using Official Code")
        detector = Detector_origin(cfg)
        detector.preprocess(cfg, [cfgs["h"],cfgs["w"]])
    
    if cfgs["dataset_group"] == "blazeit":
        video_path = os.path.join(settings.video_path,"standard_split",cfgs['full_name'],'concat',args.type,\
            '%s_%s.mp4'%(cfgs['video_name'],args.type))
    elif cfgs["dataset_group"] == "m30":
        video_path = os.path.join("%s/video_data/M30/videos","%s.mp4"%(settings.CODE_ROOT, cfgs["video_name"]))

    if cfgs["dataset_group"]=="blazeit":
        args.reid_weight = os.path.join(args.reid_weight,cfgs['video_name'],"bagtricks_R50-ibn/model_best.pth")
    else:
        args.reid_weight = os.path.join(args.reid_weight,"jackson","bagtricks_R50-ibn/model_best.pth")

    frame_sampled, resolved_tuple = get_final_tuple_from_detector\
    (tracks,args.reid_weight,video_path,detector,cfgs,args,logger)
         
    logger.info("Sampled %d frames" % len(frame_sampled))
    if args.use_filter:   
        logger.info("Filtered %d frames" % video_details.differencor)
    logger.info("Inference Time")
    logger.info(time.time()-video_details.start_time)
    logger.info("Detetor Time")
    logger.info(video_details.detector_time)
    logger.info("Differencor Time")
    logger.info(video_details.frame_differencer_time)
    logger.info("Reid Time")
    logger.info(video_details.reid_time)
    logger.info("Match Time")
    logger.info(video_details.match_time)
    logger.info("Decode Time")
    logger.info(video_details.decode_time)
    logger.info("Assign Time")
    logger.info(video_details.assign_time)

    return_tuple = parse_intervals(resolved_tuple)
        
    if args.save_result:
        results = json.dumps(return_tuple)
        f_outputs = open('./outputs/parsed_results/%s_results.json'%(cfgs['video_name']),'w')
        f_outputs.write(results)
        f_outputs.close()
        
        f_outputs = open('./outputs/parsed_results/%s_results_dict.json'%(cfgs['video_name']),'w')
        results = json.dumps(video_details.match_dict)
        f_outputs.write(results)
        f_outputs.close()

    video_details.return_tuple = return_tuple

    if args.visualize:
        paint_traj("./outputs/traj_match/",tracks)
        logger.info("Trajectory match pairs has been painted!")
        # evaluate_traj("./outputs/traj_match",tracks)
        # logger.info("Start/end bboxes have been painted.")

    traj_acc = evaluate_traj_acc(tracks)

    evaluate_query_result(args,cfgs,label_tuple_origin)
    recall_by_sampled_frame = evaluated_object_recall(video_details.frame_sampled,tuple_dict)
    
    logger.info("Trajectory Match Accuracy: ")
    logger.info(traj_acc)
    logger.info("Recall of objects: ")
    logger.info(recall_by_sampled_frame)
    if args.visualize:
        logger.info("Reid accuracy")
        logger.info(video_details.reid_acc[0]/(video_details.reid_acc[0]+video_details.reid_acc[1]))


if __name__ == "__main__":

    parser = make_parser()
    args = parser.parse_args()
    cfgs = getYaml(args.config)
    logger.add('%s.log'%(datetime.datetime.now().strftime('%Y-%m-%d_%H')))
    set_seeds(cfgs['seed'])
    clear_dir(cfgs)
    logger.info("Clear processing and visualize files")
    run_leap(cfgs,args,logger)