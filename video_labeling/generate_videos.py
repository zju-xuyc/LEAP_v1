import numpy as np
import cv2
import os

def generate_video(video_path, video_name,save_path, fps=25):

    file_list = os.listdir(video_path)
    file_list.sort(key=lambda x: int(x[-10:-4]))
    img_0 = cv2.imread(os.path.join(video_path,file_list[0]))
    
    size = (img_0.shape[1],img_0.shape[0])
    video = cv2.VideoWriter(video_name+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for file in file_list:
        image_path = os.path.join(video_path,file)
        img = cv2.imread(image_path)
        video.write(img)
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_video("/home/xyc/Datasets/M-30_mask","./videos/M-30_mask")