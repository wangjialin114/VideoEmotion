# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:07:44 2017
used for extracting the image frames from the video
## extract strategy:
##      for each barrage cluster, suppose the time t is the cluster time, we extract 3 frames at {t-delta_t, t, t+delta_t}, set delta_t=1. We choose these 3 moments for following 2 reasons:
##          1.data augmentation, we can choose the frames to train our model randomly
##          2.the frames near the cluster center should be more important
##      for cluster duration is less than 2 seconds, we will choose the begining and the end of the cluster. 
##
@author: WangJalin
"""


import numpy as np
import os
import tqdm
import subprocess
import logging
import logging.handlers


class ExtractFrame(object):
    """according the information of barrage cluster, extract the frames from the video
    """
    
    def __init__(self, cluster_file="doc2id.txt", data_dir="dataset"):
        """init method
        
        Args:
            cluster_file: the path of the barrage cluster file
        Returns:
            None
        """
        
        self.log_setup()  # log setup
        self.cluster_file = cluster_file
        self.data_dir = data_dir
        self.frame_per_cluster = 3  ## extract frame num per cluster
        self.delta_t = 1  ## extract time interval bettween the frames per cluster,{t-delta_t, t, t+delta_t}
        
    def log_setup(self):
        """log setup, log the errors when extract the image frames"""
        
        # set the logger
        LOG_FILE = 'extract_frames.log'  
        handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes = 1024*1024, backupCount = 5)   
        fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'  
        formatter = logging.Formatter(fmt)  
        handler.setFormatter(formatter)
        self.logger = logging.getLogger('extract_frames')      
        self.logger.addHandler(handler)             
        self.logger.setLevel(logging.DEBUG) 
        
    def get_duration(self, filename):
        """get the duration of the time
        
        Args:
            filename: the video filename
        Returns:
            duration: the time duration of the video
        """
        
        import re
        video_info = subprocess.Popen(["ffprobe", filename],stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        video_info  = video_info.stdout.readlines() 
        duration_info = [str(x) for x in video_info if "Duration" in str(x)]
        duration_str = re.findall(r"Duration: (.+?),", duration_info[0])   # reg match to find the duration informatio
        h, m ,s = duration_str[0].split(":")
        duration = int(h)*3600 + int(m)*60 + float(s)  # change the duration to the seconds
    
        return duration
        
    def read_cluster_file(self):
        """get the barrage cluster file, get the time information
        
        Parse the file doc2id.txt, prepare for the later video frames extraction and audio extraction
        Video_ID, Cluster_1_Time(Begin, End, Center),..., Cluster_9_Time(Begin, End, Center)
        """
        
        # read the cluster file
        f = open(self.cluster_file)
        lines = f.readlines()
        f.close()
        extract_time = {}
        # parse the time
        # the comment includes 3 numbers while the barrage line include 6 numbers
        for line in lines:
            line = line.split()
            if len(line) == 6:
                # barrage
                if extract_time.get(int(line[0])) == None:
                    extract_time[int(line[0])] = [float(line[3]), float(line[4]), float(line[5])]
                else:
                    extract_time[int(line[0])] = extract_time[int(line[0])] + [float(line[3]), float(line[4]), float(line[5])]
            elif len(line) == 3:
                # comment
                pass
            else:
                self.logger.error("Error in parse the cluster time : %s" % (line))

        self.extract_time = extract_time
    
    def extract_frame(self):
        """use the ffmpeg to extract the frame images
        
        Call the shell command to execute. Log when any error comes up
        """
        
        failed_video = []  # store the all video ids that are extracted falied
        exceed_video = []  # store the all video ids that the cluster time are exceed the duration of the video
        # traverse the extract time
        # tqdm can display the extracting progress bar, and appropriate time
        for key, value in tqdm.tqdm(self.extract_time.items()):
        #for key, value in extract_time.items():
            video_id = key
            video_name = self.data_dir + "/" + str(video_id) + "/" + str(video_id) + ".flv"
            if os.path.exists("./"+video_name) == False:
                video_name = self.data_dir + "/" + str(video_id) + "/" + str(video_id) + ".mp4"
            if os.path.exists("./"+video_name) == False:
                self.logger.error("%s not exists" %("./"+video_name))
                failed_video.append(video_id)
                continue
            # sort the time
            sort_time = np.sort(value)
            # construct the shell cmd
            if len(sort_time) != 30:
                self.logger.error("%d Only %d frames" %(video_id, len(sort_time)))
                failed_video.append(video_id)
                continue
            # get the duration of the video
            duration = self.get_duration(video_name)
            # extract the frames
            for i in range(len(sort_time)//self.frame_per_cluster):
                cluster_t = sort_time[3*i+1]  # cluster time
                for j in range(self.frame_per_cluster):
                    t = cluster_t + (j-1)*self.delta_t
                    # at the begining of the video, may no frames
                    # at the end of the video, may no frames
                    # i.e. 0.8s < t < duration - 1.5s
                    if t < 0.8:  
                        t = 0.8
                    elif t > duration + 0.5:  # the time may not be so accurate
                        self.logger.error("The cluster time %f is exceed the duration %f of the video %s" %(t, duration, video_id))
                        failed_video.append(video_id)
                        exceed_video.append(video_id)
                        break
                    elif t > duration - 1.5:  
                         t = duration - 1.5
                    shell_cmd_frame = "ffmpeg -ss "+str(t)+" -i "+video_name+" -frames:v 1 "+"frames/"+str(video_id)+"_"+str(i)+".jpg" +" -v error -y"
                    out = subprocess.getstatusoutput(shell_cmd_frame)
                    if out[1] != "":
                        failed_video.append(video_id)
                        self.logger.error("Error with %d" % video_id)
                        self.logger.error(shell_cmd_frame)
                        self.logger.error(out[1])
                        break
                    # check if the output file is empty
                    ext_filename = "./frames/" + str(video_id) + "_" + str(i) + ".jpg"
                    if os.path.exists(ext_filename) == False:
                        self.logger.error("%s is empty " % ext_filename)
                        self.logger.error(shell_cmd_frame)
                        failed_video.append(video_id)
                        break
        #print("the problem ratio: %f" %(len(set(failed_video))/len(extract_time)))
        for key in list(set(failed_video)):
            self.extract_time.pop(key)
            
        self.failed_video = failed_video
        self.exceed_video = exceed_video
    
    def record_result(self):
        """write the extract result(include successful and fialed) to the txt file"""
        
        # record the problem video id
        f = open("failed_video_id.txt",  "a")
        for x in self.failed_video:
            f.writelines(str(x) + "\n")
        f.close()
        # record the exceed video id
        f = open("exceed_video_id.txt",  "a")
        for x in self.exceed_video:
            f.writelines(str(x) + "\n")
        f.close()
        # record the extract successfully video id
        success_video_ids = [str(x) for x in self.extract_time.keys()]
        f = open("success_video_id.txt", "w")
        for x in success_video_ids:
            f.writelines(str(x) + "\n")
        f.close()

        
    def extract_process(self):
        """the main progress to execute"""
        
        self.read_cluster_file()
        self.extract_frame()
        self.record_result()

if __name__ == "__main__":
    e_f = ExtractFrame()
    e_f.extract_process()