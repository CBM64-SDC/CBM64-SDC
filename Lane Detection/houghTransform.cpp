#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src, edges;
Mat src_gray;
Mat standard_hough, probabilistic_hough;
int min_threshold = 50;
int max_trackbar = 300;

const char* standard_name = "Standard Hough Lines";
const char* probabilistic_name = "Probabilistic Hough Lines";

int s_trackbar = max_trackbar;
int p_trackbar = max_trackbar;


void help(){
  printf("\t Hough Transform to detect lines \n ");
  printf("\t---------------------------------\n ");
  printf(" Usage: ./houghTransform <image_name> \n");
}

void Standard_Hough( int, void* ){
  vector<Vec2f> s_lines;
  cvtColor( edges, standard_hough, COLOR_GRAY2BGR );

  /// 1. Use Standard Hough Transform
  HoughLines( edges, s_lines, 1, CV_PI/180, min_threshold + s_trackbar, 0, 0 );

  /// Show the result
  for( size_t i = 0; i < s_lines.size(); i++ ){
    float r = s_lines[i][0], t = s_lines[i][1];
    double cos_t = cos(t), sin_t = sin(t);
    double x0 = r*cos_t, y0 = r*sin_t;
    double alpha = 1000;

     Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
     Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
     line( standard_hough, pt1, pt2, Scalar(255,0,0), 3, LINE_AA);
   }

 imshow( standard_name, standard_hough );
}

void Probabilistic_Hough( int, void* ){
  vector<Vec4i> p_lines;
  cvtColor( edges, probabilistic_hough, COLOR_GRAY2BGR );

  /// 2. Use Probabilistic Hough Transform
  HoughLinesP( edges, p_lines, 1, CV_PI/180, min_threshold + p_trackbar, 30, 10 );

  /// Show the result
  for( size_t i = 0; i < p_lines.size(); i++ ){
     Vec4i l = p_lines[i];
     line( probabilistic_hough, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, LINE_AA);
   }

  imshow( probabilistic_name, probabilistic_hough );
}

//void calibrateImage()

// void undistortImage(image, mtx, dist){
//   imshow(cv2.undistort(image, mtx, dist, None, mtx));
// }

int main( int argc, char** argv ){
  String imageName("../data/building.jpg");
  if (argc > 1)
    imageName = argv[1];
  src = imread( imageName, IMREAD_COLOR );

  Rect myROI(0,src.rows/2,src.cols,src.rows/2);  
  src = src(myROI);

  if(src.empty()){
    help();
    return -1;
  }

   /// Pass the image to gray
   cvtColor( src, src_gray, COLOR_RGB2GRAY );

   /// Apply Canny edge detector
   Canny( src_gray, edges, 50, 200, 3 );

   /// Create Trackbars for Thresholds
   char thresh_label[50];
   sprintf( thresh_label, "Thres: %d + input", min_threshold );

   namedWindow( standard_name, WINDOW_AUTOSIZE );
   createTrackbar( thresh_label, standard_name, &s_trackbar, max_trackbar, Standard_Hough);

   namedWindow( probabilistic_name, WINDOW_AUTOSIZE );
   createTrackbar( thresh_label, probabilistic_name, &p_trackbar, max_trackbar, Probabilistic_Hough);

   /// Initialize
   Standard_Hough(0, 0);
   Probabilistic_Hough(0, 0);
   waitKey(0);
   return 0;
}