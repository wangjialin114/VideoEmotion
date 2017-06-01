## Video Modal About

#### 1. Extract Frames

##### a. extract time  

Suppose the cluster center is at $t$, we decide to extract the frame at time [t-1, t, t]. We choose these 3 time for the following 2 reasons:

1. data augmentation
2. the frame near the cluster center should be more important

##### b. image frame resize

We plan to use the pretrain cnn net vgg16, since the standard input is 224x224x3, which means that image size is 224x224. and our frames have many diffrent sizes. we need to resize our image to 224x224. The strategy is as follows:

Suppose that the frame size is $H$x$W$, $L = min{\{H,W\}}$, point $C(C_x, C_y)$ denotes  the center of the image.$x$ denotes the $W$ while $y$ is along the $H$, the left top corner point is the origin.

-  $H>W=L$,extract area: x: [0,W ], y: [$C_y-\frac{L}{2}$,$C_y+\frac{L}{2}$]
-  $L=H<W$ ,extract are: x:[$C_x-\frac{L}{2}$,$ C_x+\frac{L}{2}$], y:[0,H]

Thus, the extracted area should be $L$x$L$,then resize $L$x$L$ to 224x224.

#### 2.Pretrained Vgg16 Feature Selection

Usually we extract the fc7 feature. But for our task, like the color distribution etc low level information is more important. And extracting the semantic information from the image is very difficult.But we can still first try the fc7 first.

#### 3.Preprocessing the data

Because the parameters of  CNN part of our model is fixed, so end to end training may be time consuming. We plan to extract the CNN feature and save them to the hdf file.And may take 64GB disk for 2000 videos.

