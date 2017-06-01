# coding: utf-8
"""
Created on Sat May 20 17:07:44 2017
used for renaming the bilibili dataset
@author: WangJalin
## This script is used to read the frames.
## ##################################resize the image
## ########################store the data to the hdf file
Resize Strategy:
We plan to use the pretrain cnn net vgg16, since the standard input is 224x224x3, 
which means that image size is 224x224. and our frames have many diffrent sizes. 
we need to resize our image to 224x224. The strategy is as follows:

    Suppose that the frame size is HxW, L = min{\{H,W\}}, point C(C_x, C_y) denotes
    the center of the image.x denotes the W while y is along the H, 
    the left top corner point is the origin.

    - H>W=L,extract area: x: [0,W ], y: [C_y-\frac{L}{2},C_y+\frac{L}{2}]
    - L=H<W ,extract are: x:[C_x-\frac{L}{2}, C_x+\frac{L}{2}], y:[0,H]

Thus, the extracted area should be LxL,then resize LxL to 224x224.
"""

from PIL import Image
import numpy as np
import h5py
import os
import tqdm


class ImageToHdf(object):
    """resize the image, store the data to the hdf file"""
    
    def __init__(self, h5f):
        """init method
        Args:
            h5f: the hdf file object
        """
        self.h5f = h5f
        
    def read_resize_img(self, filename):
        """read the file and resize the file to 224*224*3
        
        Args:
            filename: the image file name
        Returns:
            a ndarray whose shape is 224*224*3
        """
        
        # check if the file exists
        if os.path.exists(self, filename):
            pass
        else:
            print("file %s not exists" % (filename))
            return None
        # read the image
        img = Image.open(filename)
        # get the width , height of th image
        width, height =  img.size
        # crop the image
        min_edge = np.min([width, height])
        # crop and resize
        if min_edge >= 224:
            left =  width//2 - min_edge//2
            right = left + min_edge
            lower = height//2 + min_edge//2
            upper = lower - min_edge
            crop_image = img.crop((left, upper, right, lower))
        else:
            left = width//2- min_edge//2
            lower = height//2 +  min_edge//2
            right = left + min_edge
            upper = lower - min_edge
            crop_image = img.crop((left, upper, right, lower))
            # resize the image
        crop_image = crop_image.resize([224, 224], resample=4)
    
        return np.asarray(crop_image)
    
    def frames_to_hdf(self):
        """ traverse all the images, resize the image, and store them to the hdf filem
        """
        # load the video id of the success extraction
        f = open("success_video_id.txt", "r")
        lines = f.readlines()
        video_id_set = [x[:-1] for x in lines]
        img_num = 30
        # Note: Be careful about this if you process the images  muliple times
        video_h5 = self.h5f.create_dataset("frames", data=np.zeros([len(video_id_set), img_num, 224*224*3]), chunks=True)
        #video_h5 = h5f["frames"]
        # read all the video
        print("%d video Frames Processing Start" % (len(video_id_set)))
        for kk in tqdm.tqdm(range(len(video_id_set))):
            video_id = video_id_set[kk]
            video_frames = np.zeros((img_num, 224*224*3))
            for i in range(img_num):
                img_name = "./frames/" + video_id + "_" + str(i) + ".jpg"
                img_array = self.read_resize_img(img_name)
                video_frames[i, :] = img_array.reshape([1, 224*224*3])
            video_h5[kk, :, :] = video_frames
        f.close()
    

if __name__ == "__main__":
    # h5py data
    h5f = h5py.File("frame_audio.h5", "w")
    f_t_h = ImageToHdf(h5f)
    f_t_h.close()

