#include <iostream>
#include <string>
#include <math.h>


#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//g++ -std=c++11 source.cpp `pkg-config --libs --cflags opencv` -o source

using namespace cv;
using namespace std;

Mat src, originalImage, srcGray, gradientX, gradientY, gradient;
Mat absoluteGradientX, absoluteGradientY, image, srcBlured, sobeled, colored, combined, edges, toShow;

string window_name = "Edge Detector";

int flag = 0;
int scale = 1;
int delta = 0;
int ddepth = CV_64FC1;//CV_16S;
int left = 100;
int right = 200;

//#########
Mat image2, standard_hough;
int max_trackbar = 50;
int s_trackbar = max_trackbar;
int min_threshold = 50;
//#########

Point oldPt1, oldPt2;

void printPicture(Mat soorah){
    uint8_t *myData = soorah.data;
    int _stride = int(soorah.step);
    for(int i=0; i < soorah.rows; ++i){
        for(int j=0; j < soorah.cols; ++j){
            //uint8_t val = myData[i + _stride + j];
            printf("%lf\t", soorah.at<double>(i, j));
        }
    }
}

void standardHough( int, void* ){
    vector<Vec2f> s_lines;
    cvtColor( edges, standard_hough, COLOR_GRAY2BGR );
    
    // Standard Hough Transform
    HoughLines( edges, s_lines, 1, CV_PI/180, min_threshold + s_trackbar, 0, 0 );
    
    /// Show the resulted lines
    bool foundleft = 0, foundright = 0;
    for( size_t i = 0; i < s_lines.size(); i++ ){
        float r = s_lines[i][0], t = s_lines[i][1];
        double cos_t = cos(t), sin_t = sin(t);
        double x0 = r*cos_t, y0 = r*sin_t;
        double alpha = 1000;
        
        Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
        Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
        if((pt1.x >= 0 || pt1.y >= 0) && !foundleft)
            foundleft = 1;
        if((pt2.x >= 0 || pt2.y >= 0) && !foundright)
            foundright = 1;
        line( standard_hough, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
    }
    
    string left, right;
    left = "Left Lane Found";
    right = "Right Lane Found";
    if (!foundleft) {
        left = "Left Lane Not Found";
    }
    if (!foundright) {
        right = "Right Lane Not Found";
    }
    putText(standard_hough, left, cvPoint(30,50), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,255), 1, CV_AA);
    putText(standard_hough, right, cvPoint(30,70), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,255), 1, CV_AA);
    //imshow( window_name, standard_hough );
    //waitKey(0);
}

Mat sobelThreshold(Mat src, uint8_t min=100, uint8_t max=200){
    //GaussianBlur to remove the noise
    GaussianBlur(src, srcBlured, Size(3,3), 0, 0, BORDER_DEFAULT);
    //Convert to grayscale
    cvtColor(src, srcGray, CV_BGR2GRAY);
    //Derivatives calculations in X and Y directions using Sobel
    //Gradient X
    Sobel(srcGray, gradientX, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    //Gradient Y
    Sobel(srcGray, gradientY, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    //Scale, calculate absolute values and convert the result to 8-bit;
    convertScaleAbs(gradientX, absoluteGradientX);
    convertScaleAbs(gradientY, absoluteGradientY);
    //Calculate the weighted sum of two arrays, (approximately!)
    addWeighted(absoluteGradientX, 0.5, absoluteGradientY, 0.5, -75, gradient);
    
    vector<vector<uint8_t> > img;
    img.resize(gradient.rows);
    for(int i=0; i<gradient.rows; ++i){
        vector<uint8_t> temp;
        temp.resize(gradient.cols);
        uint8_t* p = gradient.ptr<uint8_t>(i);
        for(int j=0; j<gradient.cols; ++j){
            //uint8_t val = myData[i + _stride + j];
            if(p[j] >= min && p[j] <= max)
                temp[j] = 255;
            else
                temp[j] = 0;
        }
        img[i] = temp;
    }
    Mat sobeled((int)img.size(), (int)img[0].size(), CV_8UC1);
    for(int i=0; i<sobeled.rows; ++i){
        for(int j=0; j<sobeled.cols; ++j){
            sobeled.at<uint8_t>(i,j) = img[i][j];
        }
    }
    cvtColor(sobeled, sobeled, CV_GRAY2BGR);
    return sobeled;
    
    // imshow(window_name, myImage);
    // waitKey(0);
}

Mat colorThreshold(Mat src, uint8_t min = 100, uint8_t max = 200) {
    cvtColor(src, src, CV_BGR2HLS);
    Mat channels[3];
    split(src, channels);
    Mat srcHLS = channels[1];
    vector<vector<uint8_t> > img;
    img.resize(srcHLS.rows);
    for(int i=0; i<srcHLS.rows; ++i){
        vector<uint8_t> temp;
        temp.resize(srcHLS.cols);
        uint8_t* p = srcHLS.ptr<uint8_t>(i);
        for(int j=0; j<srcHLS.cols; ++j){
            if(p[j] > min && p[j] <= max)
                temp[j] = 255;
            else
                temp[j] = 0;
        }
        img[i] = temp;
    }
    Mat resColor((int)img.size(), (int)img[0].size(), CV_8UC1);
    for(int i=0; i<resColor.rows; ++i){
        for(int j=0; j<resColor.cols; ++j){
            resColor.at<uint8_t>(i,j) = img[i][j];
        }
    }
    return resColor;
}

int main(int argc, char** argv){
    //src = imread(argv[1]);
    string path = "/Users/mohammedamarnah/Desktop/SDCProject/CBM64-SDC/Lane Detection/Lane Detection/Samples/";
    //string path2 = argv[1];
    VideoCapture cap(path+"Challenge.mp4");
    
    if(!cap.isOpened()){
        cout << "Error, not a valid video!" << endl;
        exit(1);
    }
    while(1){
        double time = clock();
        
        cap >> src;
        if(src.empty())
            break;
        //Cut the upper part of the image
        Rect myROI(src.cols-(src.cols*0.9), src.rows/2, src.cols-(src.cols*0.1), src.rows/2);
        src = src(myROI);
        
        if(!src.data)
            return -1;
        originalImage = src;
        
        //Apply Sobel Threshold
        sobeled = sobelThreshold(src, 150, 255);
        
        //Apply Color Threshold
        colored = colorThreshold(src, 160, 255);
        
        //Apply combined Threshold
        combined = colorThreshold(sobeled, 160, 255);
        
        //Apply Canny edge detector
        //Canny(sobeled, edges, 50, 200, 3 );
        
        //Standard HoughTransform
        //standardHough(0, 0);
        
        if (flag == 1)
            toShow = sobeled;
        else if (flag == 2)
            toShow = colored;
        else
            toShow = combined;

        // Compute the time elapsed in processing one frame
        double timeElapsed = clock() - time;
        timeElapsed = 1/(timeElapsed/CLOCKS_PER_SEC);
        putText(toShow, to_string(timeElapsed)+" FPS", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255,255,255), 1, CV_AA);
        
        //Show the result
        namedWindow(window_name, WINDOW_AUTOSIZE);
        imshow(window_name, toShow);
        
        char c = (char)waitKey(25);
        if(c==27) {
            break;
        } else if (c == 32) {
            flag = (flag + 1)%3;
        }
    }
}
