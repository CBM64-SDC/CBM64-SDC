//
//  ImageProcessing.cpp
//  TrafficRecognition
//
//  Created by Mohammed Amarnah on 12/11/17.
//  Copyright © 2017 Mohammed Amarnah. All rights reserved.
//

#include "ImageProcessing.h"

cv::Mat imageProcessing::filterContours(cv::Mat &segImg) {
    cv::Mat ret = segImg.clone();
    
    // Create the structuring element for the erosion and dilation
    cv::Mat structElt = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(4, 4));
    
    // Apply the dilation
    cv::dilate(ret, ret, structElt);
    // Threshold the image
    cv::threshold(ret, ret, 254, 255, CV_THRESH_BINARY);
    
    // Find the contours of the objects
    std::vector< std::vector< cv::Point > > contours;
    std::vector< cv::Vec4i > hierarchy;
    cv::findContours(ret, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // Filled the objects
    cv::Scalar color(255, 255, 255);
    cv::drawContours(ret, contours, -1, color, CV_FILLED, 8);
    
    // Apply some erosion on the destination image
    cv::erode(ret, ret, structElt);
    
    // Noise filtering via median filtering
    for (int i = 0; i < 5; ++i)
        cv::medianBlur(ret, ret, 5);
    
    return ret;
}

// Function to remove ill-posed contours
void imageProcessing::eliminateBad(std::vector< std::vector< cv::Point > >& contours, const cv::Size size_image, const long int areaRatio, const double lowAspectRatio, const double highAspectRatio) {
    
    for (auto it = contours.begin(); it != contours.end(); /*We remove some part of the vector - DO NOT INCREMENT HERE*/) {
        // Find a bounding box to compute around the contours
        const cv::Rect bound_rect = cv::boundingRect(cv::Mat(*it));
        // Compute the aspect ratio
        const double ratio = static_cast<double> (bound_rect.width) / static_cast<double> (bound_rect.height);
        const long int areaRegion = bound_rect.area();
        
        // Check the inconsistency
        if ((areaRegion < size_image.area() / areaRatio) || ((ratio > highAspectRatio) || (ratio < lowAspectRatio)))
            contours.erase(it);
        else
            ++it;
    }
}

// Compute the distance between the edge points (po and pf), with the current point pc
float imageProcessing::distance(const cv::Point& po, const cv::Point& pf, const cv::Point& pc) {
    // Cast into float to compute right distances
    const cv::Point2f po2f(static_cast<float> (po.x), static_cast<float> (po.y));
    const cv::Point2f pf2f(static_cast<float> (pf.x), static_cast<float> (pf.y));
    const cv::Point2f pc2f(static_cast<float> (pc.x), static_cast<float> (pc.y));
    
    // In this function, we will compute the altitude of the triangle form by the two points of the convex hull and the one of the contour.
    // It will allow us to remove points far of the convex conserving a degree of freedom
    
    // Compute the three length of each side of the triangle
    // a will be the base too
    const float a = std::sqrt(std::pow(pf2f.x - po2f.x, 2.00) + std::pow(pf2f.y - po2f.y, 2.00));
    // Compute the two other sides
    const float b = std::sqrt(std::pow(pc2f.x - po2f.x, 2.00) + std::pow(pc2f.y - po2f.y, 2.00));
    const float c = std::sqrt(std::pow(pf2f.x - pc2f.x, 2.00) + std::pow(pf2f.y - pc2f.y, 2.00));
    
    // Compute S which is the perimeter of the triangle divided by 2
    const float s = (a + b + c) / 2.00;
    // Compute the area of the triangle
    const float area = std::sqrt(s * (s - a) * (s - b) * (s - c));
    // Compute the altitude
    return 2.00f * area / a;
}

vector<vector<cv::Point> > imageProcessing::contourThreshold(vector<vector<cv::Point> > &hullContours, vector<vector<cv::Point> > &contours, const double dist) {
    vector<vector<cv::Point> > finalContours;
    cv::Point currentHullPoint, contourPoint, nextHullPoint;
    
    for (size_t contourIDX = 0; contourIDX < contours.size(); contourIDX++) {
        int hullIDX = 0;
        currentHullPoint = hullContours[contourIDX][hullIDX];
        contourPoint = contours[contourIDX][0];
        
        while(currentHullPoint != contourPoint) {
            hullIDX++;
            currentHullPoint = hullContours[contourIDX][hullIDX];
        }
        
        hullIDX = ((hullIDX - 1) + (int)hullContours[contourIDX].size()) % (int)hullContours[contourIDX].size();
        nextHullPoint = hullContours[contourIDX][hullIDX];
        
        vector<cv::Point> good;
        for (size_t i = 0; i < contours[contourIDX].size(); i++) {
            contourPoint = contours[contourIDX][i];
            if (distance(currentHullPoint, nextHullPoint, contourPoint) < dist) {
                good.push_back(contourPoint);
            }
            if (nextHullPoint == currentHullPoint) {
                currentHullPoint = hullContours[contourIDX][hullIDX];
                hullIDX = ((hullIDX - 1) + (int)hullContours[contourIDX].size()) % (int)hullContours[contourIDX].size();
                nextHullPoint = hullContours[contourIDX][hullIDX];
            }
        }
        finalContours.push_back(good);
    }
    return finalContours;
}

void imageProcessing::contourExtraction(cv::Mat &img, vector<vector<cv::Point> > &contourPoints, bool cw) {
    // allocate 2 arrays for getting the output of cv::findContours in
    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;
    
    // OpenCV's findContours function to determine the contours in the image
    cv::findContours(img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    eliminateBad(contours, img.size());
    
    // Finding the convex hull points
    vector<vector<cv::Point> > hullPoints(contours.size());
    
    for (int it = 0, it_hull = 0; it < contours.size(); it++, it_hull++) {
        cv::convexHull(cv::Mat(contours[it]), hullPoints[it_hull], cw);
    }
    
    // Extract the contours from the convex hull points
    contourPoints = contourThreshold(hullPoints, contours);
    
    for (int i = 0; i < contourPoints.size(); i++) {
        for (int j = 0; j < contourPoints[i].size(); j++) {
            img.at<uint8_t>(contourPoints[i][j]) = 0;
        }
    }
    return;
}
