# About Codes

all the codes are writen in python.

## Environment

- Python3.5

- tensorflow 1.1

- ffmpeg

- librosa

  and for other python packages, you can use `pip install package-name ` whenever there comes up a `ImportError`.

## Code Guide

#### Rename the video file

rename_file.py:  because the video files' names are all in characters,  it is necessary to rename the file according to their video id.

***this should be already been done, you can just leave it alone.***

#### Extract the image frames and the audio clips from the video file

extract_audio.py: extract the audio clips from the video file.

extract_frame.py: extract the image frames from the video file and will import the extract_audio to extract the audio clips, also log  the errors when extracting. 

***in this step, you can just type `python extract_frame.py` for extracting images and audio clips.***

#### Preprocess the data

extract_fc7.py: extract the fc_7 feature of the image frames ad save them to the hdf file.

frame_audio_to_h5.py: extract the spectrum of the audio clips and load the image and save them to the hdf file

#### Classification

audio_clf_conv4.py: audio classification CNN class

video_clf_lstm.py: video classification CNN+LSTM class

## TODO

- refractor  the code



