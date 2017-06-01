# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:07:44 2017
used for renaming the bilibili dataset
@author: WangJalin
"""


import os  
import re

class RenameFile(object):
    """used for renaming the bilibili video file name to the video id(i.e. the dir_name of the video)
    
    For video that just includes one video, just rename the video
    For video that include multiple videos, just rename the first video
    """
    
    def __init__(self, print_name=False):
        """init method
        
        Args:
            print_name: decide if print the video name
        """
        
        self.process_video_name(print_name)
        
    def get_filename(self, filenames):
        """get the filename for the multiple videos of one video id
        
        for video that has more one videos, we can found that each video name is in format as 
        [filename + number + "、" + sub_name]. And the video names are varied by the number 
        and the sub_name.What we want to do is to extract the filename
        
        Args:
            filenames: the video files' names that are more than 1
        
        Returns:
            filename: the common filename that multiple video names share
        """
        
        for i in range(min(len(filenames[0]), len(filenames[1]))):
            if filenames[0][:i+1] != filenames[1][:i+1]:
                filename = filenames[0][0:i-1]
                return filename


    def process_video_name(self, print_name=False):
        """rename the video, change the filename to the video_id"""   
        cur_dir = os.getcwd()  
        cnt = 0
        
        for parent, dirnames, filenames in os.walk(cur_dir):   
            if cnt == 0:  # pass the cur_dir
                cnt = 1
                continue
            else:
                file_num = len(os.listdir(parent))
                # coount the number of the files, 
                # if it is more than 7, than there should be more than 2 videos
                if file_num > 7:
                    # get the video_id from the directory name
                    video_id = re.findall(r"\d+", parent)[-1]
                    # get the video names
                    video_names = []
                    for filename in filenames:
                        if ".flv" in filename or ".mp4" in filename:
                            video_names.append(filename)
                    video_name = get_filename(video_names)
                    # record the video name
                    record = str(video_id) + " " + video_name
                    if print_name:
                        print(record)
                    # traverse all the video file, find the first video file
                    # rename the first video file
                    for filename in filenames:
                        split_part = filename[len(video_name):]
                        if ".flv" in split_part or ".mp4" in split_part:
                            video_seq = re.findall(r"-(\d+)", split_part)
                            if len(video_seq) > 0 and video_seq[0] == str(1):  # judge if it is the first video file
                                # rename the video
                                new_name = str(video_id) + split_part[-4:]
                                os.rename(os.path.join(parent, filename), os.path.join(parent, new_name)) 
                                break
                else:
                    # only one video
                    video_id = re.findall(r"\d+", parent)[-1]
                    for filename in filenames:
                        if ".flv" in filename or ".mp4" in filename:
                            # record the video name
                            record = str(video_id) + " " + filename[:-4]
                            if print_name:
                                print(record)
                            # rename the file
                            new_name = str(video_id) + filename[-4:]
                            os.rename(os.path.join(parent, filename), os.path.join(parent, new_name))  
                            break

if __name__ == "__main__":
    rn = RenameFile()
    rn.process_video_name()