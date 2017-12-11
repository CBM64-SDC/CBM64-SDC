//
//  ImageProcessing.hpp
//  TrafficRecognition
//
//  Created by Mohammed Amarnah on 12/11/17.
//  Copyright © 2017 Mohammed Amarnah. All rights reserved.
//

#ifndef ImageProcessing_h
#define ImageProcessing_h

#include "ColorProcessing.h"

class imageProcessing {
public:
    // Constructor and Destructor
    imageProcessing() {}
    ~imageProcessing() {}
    
    // Fills the shape of the traffic sign
    cv::Mat filterContours(cv::Mat &segImg);
    
    // Eliminate (bad) points in the image
    void eliminateBad(vector<vector<cv::Point> > &contours, const cv::Size sizeImage, const long int areaRatio = 1500, const double lowAspectRatio = 0.5, const double highAspectRatio = 1.3);
    
    // Compute the distance between the edge points (po and pf), with the current point pc
    float distance(const cv::Point& po, const cv::Point& pf, const cv::Point& pc);
    
    // Removes the inconsistent points inside each contour
    vector<vector<cv::Point> > contourThreshold(vector<vector<cv::Point> > &hullContours, vector<vector<cv::Point> > &contours, const double dist = 2.0);
    
    // Eliminates bad things and extracts the contours using convex hull
    void contourExtraction(cv::Mat &img, vector<vector<cv::Point> > &contourPoints, bool cw = 0);
};

#endif /* ImageProcessing_h */
