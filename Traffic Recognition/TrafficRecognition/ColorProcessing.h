//
//  ColorProcessing.hpp
//  TrafficRecognition
//
//  Created by Mohammed Amarnah on 12/11/17.
//  Copyright © 2017 Mohammed Amarnah. All rights reserved.
//

#ifndef ColorProcessing_h
#define ColorProcessing_h

#include "libraries.h"
using namespace std;

class colorProcessing {
public:
    // Constructor and Destructor
    colorProcessing() {}
    ~colorProcessing() {}
    
    // Color space conversion to improved HLS
    cv::Mat BGR2IHLS(cv::Mat src);
    
    // Threshold based on the IHLS color space
    cv::Mat colorThreshold(cv::Mat img, bool color, uint8_t huemax, uint8_t huemin, uint8_t satmin);
    
    // Theta computation
    inline double theta(double r, double g, double b) {
        return acos((r - (g * 0.5) - (b * 0.5)) / sqrtf((r * r) + (g * g) + (b * b) - (r * g) - (r * b) - (g * b)));
    }
    
    // Hue computation -- H = θ if B <= G -- H = 2 * pi − θ if B > G
    inline double getHue(double r, double g, double b) {
        return (b <= g) ? (theta(r, g, b) * 255.f / (2.f * M_PI)) : (((2.f * M_PI) - theta(r, g, b)) * 255.f / (2.f * M_PI));
    }
    
    // Luminance computation -- L = 0.210R + 0.715G + 0.072B
    inline double getLum(double r, double g, double b) {
        return (0.210f * r) + (0.715f * g) + (0.072f * b);
    }
    
    // Saturation computation -- S = max(R, G, B) − min(R, G, B)
    inline double getSat(double r, double g, double b) {
        return (max(r, max(g, b)) - min(r, min(g, b)));
    }
};

#endif /* ColorProcessing_h */
