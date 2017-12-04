#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <string>
#include <math.h>

//g++ -std=c++11 source.cpp `pkg-config --libs --cflags opencv` -o source

using namespace cv;
using namespace std;

Mat src, originalImage, srcGray, gradientX, gradientY, gradient;
Mat absoluteGradientX, absoluteGradientY, image;
//int ddepth, scale, delta;
string window_name = "Simple Edge Detector";

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
//vector<vector<int>> image;

void printPicture(Mat soorah){
	uint8_t *myData = soorah.data;
	int _stride = soorah.step;
	for(int i=0; i < soorah.rows; ++i){
		for(int j=0; j < soorah.cols; ++j){
			uint8_t val = myData[i + _stride + j];
			printf("%lf\t", soorah.at<double>(i, j));
		}
	}
}

void Standard_Hough( int, void* ){
  vector<Vec2f> s_lines;
  cvtColor( image2, standard_hough, COLOR_GRAY2BGR );

  /// 1. Use Standard Hough Transform
  HoughLines( image2, s_lines, 1, CV_PI/180, min_threshold + s_trackbar, 0, 0 );

  /// Show the result
  bool foundleft = 0, foundright = 0;
  for( size_t i = 0; i < s_lines.size(); i++ ){
    float r = s_lines[i][0], t = s_lines[i][1];
    double cos_t = cos(t), sin_t = sin(t);
    double x0 = r*cos_t, y0 = r*sin_t;
    double alpha = 1000;

     Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
     Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
     //cout << pt1.x << " " << pt1.y << endl << pt2.x << " " << pt2.y << endl << endl << endl;
     if ((pt1.x >= 0 || pt1.y >= 0) && !foundleft)
     	foundleft = 1;
     foundright = (pt2.x >= 0 || pt2.y >= 0);
     line( standard_hough, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
 	}

 	if (foundleft)
 		cout << "Left Lane Found!" << " ";
 	else cout << "LACOSA KILLED ME :(" << " ";

 	if (foundright)
 		cout << "Right Lane Found!" << endl;
 	else
 		cout << "LACOSA KILLLLLED ME! :(" << endl;

	//imshow( window_name, standard_hough );
 	//waitKey(0);
}

Mat SobelThreshold(Mat imageX, uint8_t min=100, uint8_t max=200){

	vector<vector<uint8_t> > img;
	img.resize(imageX.rows);
	for(int i=0; i<imageX.rows; ++i){
		vector<uint8_t> temp;
		temp.resize(imageX.cols);
		uint8_t* p = imageX.ptr<uint8_t>(i);
		for(int j=0; j<imageX.cols; ++j){
			//uint8_t val = myData[i + _stride + j];
			if(p[j] >= min && p[j] <= max)
				temp[j] = 255;
			else
				temp[j] = 0;
		}
		img[i] = temp;
	}
	Mat myImage(img.size(), img[0].size(), CV_8UC1);
	for(int i=0; i<myImage.rows; ++i){
		for(int j=0; j<myImage.cols; ++j){
			myImage.at<uint8_t>(i,j) = img[i][j];
		}
	}
	return myImage;
	// imshow(window_name, myImage);
	// waitKey(0);

}

int main(int argc, char** argv){
	//src = imread(argv[1]);
	VideoCapture cap(argv[1]);

	if(!cap.isOpened()){
		cout << "Error" << endl;
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
		//GaussianBlur to remove the noise
		GaussianBlur(src, src, Size(3,3), 0, 0, BORDER_DEFAULT);
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
		//###
		image = SobelThreshold(gradient);
		//Apply Canny edge detector
		Canny(image, image2, 50, 200, 3 );
		// Create Trackbars for Thresholds
   		//char thresh_label[50];
		//createTrackbar( thresh_label, window_name, &s_trackbar, max_trackbar, Standard_Hough);
		//
		Standard_Hough(0, 0);
		//threshold(gradient);
		//Show the result
		namedWindow(window_name, WINDOW_AUTOSIZE);
		double timeElapsed = clock() - time;
		timeElapsed = 1/(timeElapsed/CLOCKS_PER_SEC);

		putText(standard_hough, to_string(timeElapsed)+" FPS", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,255), 1, CV_AA);
		imshow(window_name, standard_hough);
		char c = (char)waitKey(25);
		if(c==27)
			break;
	}
}