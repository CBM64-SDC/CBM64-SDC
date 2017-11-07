#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<stdio.h>
using namespace cv;


int main(int argc, char** argv) {

    if (argc < 3) {
        printf("Insufficient number parameters were passed to the file.\n");
        printf("Compliation terminated\n");
        return 0;
    }

    CascadeClassifier myCar(argv[1]); // Add the xml file you prased to the main function
    VideoCapture myVideo(argv[2]); //Add the video you parsed to the main function.

    if (!myVideo.isOpened()) {
        printf("Sorry, we couldn't open your video file\n");
        return -1;
    }

    Mat Frame;
    Mat GFrame;
    

    namedWindow("output", CV_WINDOW_AUTOSIZE);

 
    while (1) {
        myVideo.read(Frame);
        cvtColor(Frame, GFrame, CV_BGR2GRAY);
        vector<Rect> cars;
        myCar.detectMultiScale(GFrame, cars, 1.1, 1, CV_HAAR_DO_CANNY_PRUNING, Size(0, 0), GFrame.size()); 
        // In the fourth paramter above we can try CV_HAAR_DO_CANNY_PRUNING or CV_HAAR_SCALE_IMAGE.


        for (int i = 0; i < cars.size();i++) {

            Point pt1(cars[i].x + cars[i].width, cars[i].y + cars[i].height);
            Point pt2(cars[i].x, cars[i].y);

            
            rectangle(Frame, pt1, pt2, Scalar(0, 0, 255, 0), 2, 8 ,0);
        }


        
        imshow("output", Frame);
        waitKey(33);
    }

    return 0;
}