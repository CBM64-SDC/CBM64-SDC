//
//  ColorProcessing.cpp
//  TrafficRecognition
//
//  Created by Mohammed Amarnah on 12/11/17.
//  Copyright © 2017 Mohammed Amarnah. All rights reserved.
//

#include "ColorProcessing.h"

// Convert from BGR to Improved HLS
cv::Mat colorProcessing::BGR2IHLS(cv::Mat src) {
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
cv::Mat colorProcessing::colorThreshold(cv::Mat img, bool color, uint8_t huemax, uint8_t huemin, uint8_t satmin) {
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
