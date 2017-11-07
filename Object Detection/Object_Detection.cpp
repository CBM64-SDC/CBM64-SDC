#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<stdio.h>
using namespace cv;
using namespace std;
int main() {
    
    CascadeClassifier myCar("temp/vehicle_detection_haarcascades/cars.xml");
    VideoCapture myVideo("project_video.mp4");


    if (!myVideo.isOpened()) {
        printf("Sorry, we couldn't open your video file\n");
        return -1;
    }

    Mat Frame;
    Mat GFrame;

    namedWindow("output", CV_WINDOW_AUTOSIZE);
    while (true) {
        myVideo.read(Frame);
        // myVideo >> captureFrame;
        cvtColor(Frame, GFrame, CV_BGR2GRAY);
        vector<Rect> cars;
        myCar.detectMultiScale(GFrame, cars, 1.1, 3, CV_HAAR_SCALE_IMAGE, Size(60, 60));

        for (int i = 0; i<cars.size();i++) {
            Point pt1(cars[i].x + cars[i].width, cars[i].y + cars[i].height);
            Point pt2(cars[i].x, cars[i].y);

            rectangle(Frame, pt1, pt2, cvScalar(0, 255, 0,0 ), 1, 0 ,0);
        }
        waitKey(33);
        imshow("output", Frame);
    }
}