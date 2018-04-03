#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 500;
int ratio = 3;
int kernel_size = 3;
const char* window_name = "Edge Map";

static void CannyThreshold(int, void*){
    //![reduce_noise]
    /// Reduce noise with a kernel 3x3
    blur(src_gray, detected_edges, Size(3,3));
    /// Canny detector
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
    /// Using Canny's output as a mask, we display our result
    //![fill]
    dst = Scalar::all(0);
    //![fill]
    src.copyTo( dst, detected_edges);
    
    imshow( window_name, dst );
}

int main( int, char** argv ){
    src = imread(argv[1], IMREAD_COLOR); // Load an image
    
    if( src.empty() )
    { return 0; }
    
    /// Create a matrix of the same type and size as src (for dst)
    dst.create( src.size(), src.type() );
    
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    
    namedWindow( window_name, WINDOW_AUTOSIZE );
    
    /// Create a Trackbar for user to enter threshold
    createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
    
    /// Show the image
    CannyThreshold(0, 0);
    
    /// Wait until user exit program by pressing a key
    waitKey(0);
    
    return 0;
}