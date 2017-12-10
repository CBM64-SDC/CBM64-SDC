//
//  main.cpp
//  TrafficRecognition
//
//  Created by Mohammed Amarnah on 12/7/17.
//  Copyright © 2017 Mohammed Amarnah. All rights reserved.
//

#include <iostream>
#include <math.h>
#include <string>
#include <memory.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

// Theta computation
inline float theta(double r, double g, double b) {
    return acos((r - (g * 0.5) - (b * 0.5)) / sqrtf((r * r) + (g * g) + (b * b) - (r * g) - (r * b) - (g * b)));
}

// Hue computation -- H = θ if B <= G -- H = 2 * pi − θ if B > G
inline float getHue(double r, double g, double b) {
    return (b <= g) ? (theta(r, g, b) * 255.f / (2.f * M_PI)) : (((2.f * M_PI) - theta(r, g, b)) * 255.f / (2.f * M_PI));
}

// Luminance computation -- L = 0.210R + 0.715G + 0.072B
inline float getLum(double r, double g, double b) {
    return (0.210f * r) + (0.715f * g) + (0.072f * b);
}

// Saturation computation -- S = max(R, G, B) − min(R, G, B)
inline float getSat(double r, double g, double b) {
    return (max(r, max(g, b)) - min(r, min(g, b)));
}

// Convert from BGR to Improved HLS
cv::Mat BGR2IHLS(cv::Mat& src) {
    cv::Mat out;
    // Making sure the image has 3 channels
    CV_Assert(src.channels() == 3);
    
    out = src.clone();
    for (auto it = out.begin<cv::Vec3b>(); it != out.end<cv::Vec3b>(); it++) {
        cv::Vec3b bgr = (*it);
        (*it)[0] = getSat(double(bgr[0]),double(bgr[1]),double(bgr[2]));
        (*it)[1] = getLum(double(bgr[0]),double(bgr[1]),double(bgr[2]));
        (*it)[2] = getHue(double(bgr[0]),double(bgr[1]),double(bgr[2]));
    }
    return out;
}

// Thresholding either on blue or red channels (0: Blue, 1: Red)
cv::Mat colorThreshold(cv::Mat img, bool color, uint8_t huemax, uint8_t huemin, uint8_t satmin) {
    cv::Mat out;
    CV_Assert(img.channels() == 3);
    out.create(img.size(), CV_8UC1);
    
    for (int i = 0; i < img.rows; i++) {
        const uchar *imgData = img.ptr<uchar>(i);
        uchar *outData = out.ptr<uchar>(i);
        for (int j = 0; j < img.cols; j++) {
            uchar s = *imgData++;
            // Although l is not being used, and we can replace it
            // with (*imgData++) just to get to h segment
            // but we left it like this for the sake of readability
            uchar l = *imgData++;
            uchar h = *imgData++;
            //*outData++ = ((h > huemin && h < huemax) && s > satmin) ? 255 : 0;
            if (!color) {
                *outData++ = ((h > huemin && h < huemax) && s > satmin) ? 255 : 0;
            } else {
                *outData++ = ((h > huemax || h < huemax) && s > satmin) ? 255 : 0;
            }
        }
    }
    return out;
}

vector<vector<cv::Point> > contourThreshold(vector<vector<cv::Point> > &hullContours, vector<vector<cv::Point> > &contours) {
    vector<vector<cv::Point> > finalContours;
    
    return finalContours;
}

void contourExtraction(cv::Mat &img, vector<vector<cv::Point> > &contourPoints, bool cw = 0) {
    // allocate 2 arrays for getting the output of cv::findContours in
    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;
    
    // OpenCV's findContours function to determine the contours in the image
    cv::findContours(img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    // Finding the convex hull points
    vector<vector<cv::Point> > hullPoints(contours.size());
    
    for (int it = 0, it_hull = 0; it < contours.size(); it++, it_hull++) {
        cv::convexHull(cv::Mat(contours[it]), hullPoints[it_hull], cw);
    }
    
    // Extract the contours from the convex hull points
    contourPoints = contourThreshold(hullPoints, contours);
    return;
}

int main() {
    double time = clock();
    
    cv::Mat input = cv::imread("/Users/mohammedamarnah/Desktop/SDCProject/CBM64-SDC/Traffic Signs Recognition/data/stop.jpg");
    input = BGR2IHLS(input);
    
    // RED: 1, 15, 240, 25
    // BLUE: 0, 163, 134, 39
    cv::Mat img1 = colorThreshold(input, 1, 15, 240, 150);
    //cv::Mat img2 = colorThreshold(input, 0, 163, 134, 39);
    
    cv::Mat out = img1;
    //cv::bitwise_and(img1, img2, out);
    cout << "Time Elapsed: " << (clock() - time) / CLOCKS_PER_SEC << endl << endl;
    cv::imshow("output", out);
    cv::waitKey(0);
}
