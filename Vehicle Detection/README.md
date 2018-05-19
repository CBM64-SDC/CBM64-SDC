# Single Shot Multibox Detector (SSD) for Vehicle Detection and Tracking

[//]: # (Image References)
[image1]: ./images/img1.png
[image2]: ./images/img2.png
[image3]: ./out_images/test1_out.jpg
[image4]: ./out_images/test2_out.jpg
[image5]: ./out_images/test3_out.jpg
[image6]: ./out_images/test4_out.jpg
[image7]: ./out_images/test5_out.jpg
[image8]: ./out_images/test6_out.jpg
[image9]: ./out_images/test7_out.jpg
[image10]: ./out_images/test8_out.jpg
[image11]: ./out_images/test9_out.jpg

SSD | Single Shot MultiBox Detector
---
### Real time object detection using deep learning
Computer vision and Image Processing algorithms have been used in solving object detection problems since a very long time. Until very recently, there wasn't any way to detect objects (or maybe there was, but it was with terrible performance and accuracy) in an image.

But with the rise of the deep learning era, and more specifically when [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) won the 2012 ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) it changed the way people saw object detection problems. Deep learning architectures surpassed many classical and traditional computer vision and image processing algorithms. Nowadays deep learning models are much better than humans in image classification and object detection tasks.

#### Before SSD (R-CNN, YOLO, etc.)
Before SSD was published, there were several architectures that was used for object detection, and they worked really well in most cases.

A few years ago, a special kind of Convolutional Neural Networks (CNNs) was published that was able to locate and detect objects in images. The output of the network would usually be a set of bounding boxes that closely matched each of the objects detected along with a class for each box.

![image1]

After the R-CNN came the Fast R-CNN and the Faster R-CNN architectures. Each one of them propsed improvements to the problems of its predecessor. But all of these architectures had mutual problems that weren't solved like the time it needed to train a huge amount of data even on a GPU, or that the training happens in multiple phases. Also, one of the biggest problems of all these architectures, is that they were so slow in real time. Meaning that when predictions were run, it would take a lot of time to predict the output.

#### And then came the SSD
In November 2016, C. Szegedy et al. published the paper about the [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) and they reached amazing results. They scored over 74% mAP (mean Average Precision) at 59 frames per second on standard datasets such as PascalVOC and COCO.

![image2]

As it is clear in the architecture image above, the work on the SSD architecture was inspired by the VGG-16 archicture (and this shows the inevitable powers of transfer learning).

Vehicle Detection using SSD
---
#### 1. Model Architecture
So instead of using the traditional computer vision and image processing techniques that was propsed to do this project, I decided to go for a deep learning approach in implementing a solution for this problem.
SSD used the VGG-16 as a base for the architecture, in addition to a set of auxiliary convolution layers that were added to enable feature extraction on multiple scales and progressively decrease the size of the input to each subsequent layer.

#### 2. Model Training
Training the SSD takes a lot of time on normal computers. For effecient training you need to use a computer with multiple GPUS (note that prediction does not need a lot of time because of how the architecture is built). So instead of training the model locally, I used pretrained weights. The network was trained on the PascalVOC dataset which consisted of 21 classes one of which were the car.

#### 3. Running predictions and output decoding
After loading the model, I created the pipeline that processes each frame in the video stream. Each frame gets passed to the ``process_frame(img)`` function and then it is resized, and some preprocessing is run on it.
After that, we run the predictions on it, and use the ``BBoxUtil`` to decode the outputs from the network. After that, further decoding is done using the function shown below. This function extracts the boxes that passes a certain threshold and then returns those boxes along with there confidence scores and labels (which will always be a car).

```
def decode_output(out, thresh=0.6):
    out_label  = out[0][:,0]
    conf_score = out[0][:,1]
    x1         = out[0][:,2]
    y1         = out[0][:,3]
    x2         = out[0][:,4]
    y2         = out[0][:,5]
    
    indices = []
    for i in range(len(conf_score)):
        if conf_score[i] >= thresh and out_label[i] == 7:
            indices.append(i)
    
    return conf_score[indices], out_label[indices].tolist(), x1[indices], y1[indices], x2[indices], y2[indices]
``` 

#### 4. Running the pipeline on test images
Before I ran the pipeline on the videos, I wanted to test it on the images provided. I also ran the pipeline on test images that I downloaded from the internet. Here are some of the results

![image9]
![image6]
![image11]

#### 5. Running the pipeline on a video stream
I ran the pipeline on the videos provided in the github repo along with two other videos. You can find all the results in the [videos folder](https://github.com/mohammedamarnah/vehicle-detection/tree/master/videos/) of this repo.

Future Work
---
Many things could be added to this project to make it better. One thing is to use one of the masking/segmentation architectures, and instead of drawing boxes on the vehicles detected the architecture could output the pixels that it thinks belongs to a car. This would remove a lot of the false positives in the detection process.

Also, I will be taking this architecture and implementing it on a RaspberryPi with a camera and then I will run the pipeline in real time and record a video of it detecting vehicles here in my country. I will be uploading the videos very soon on youtube.

Acknowledgements
---
The SSD implementation that was used in this project was done by rykov8 and can be found in this [repo](https://github.com/rykov8/ssd_keras). Also all the work done initially on the SSD and the research paper can be found [here](https://arxiv.org/abs/1512.02325) I really recommend you read the research since it really strengthens your knowledge in deep learning a lot.
