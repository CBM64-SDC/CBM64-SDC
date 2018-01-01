#include <iostream>
#include <string>
#include <math.h>
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//g++ -std=c++11 source.cpp `pkg-config --libs --cflags opencv` -o source

using namespace cv;
using namespace std;

Mat src, originalImage;
Mat image, sobeledX, sobeledY, colored, perspectiveTransformed, combined, dirThreshold, magThreshold, hlsed, combinedColor;
Mat edges, toShow;

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

//######### Trackbar thingies #########
int g_slider_position = 0;
int g_run = 1, g_dontset = 0;
VideoCapture cap;
//######### #########

void onTrackbarSlide(int pos, void *){
    cap.set(CAP_PROP_POS_FRAMES, pos);
    if(!g_dontset)
        g_run = 1;
    g_dontset = 0;
}

void printPicture(Mat soorah){
    for(int i=0; i<soorah.rows; ++i){
        for(int j=0; j<soorah.cols; ++j){
            printf("%d\t", soorah.at<uint8_t>(i,j));
        }
        printf("\n");
    }

    //Not working method

    // uint8_t *myData = soorah.data;
    // int _stride = int(soorah.step);
    // for(int i=0; i < soorah.rows; ++i){
    //     for(int j=0; j < soorah.cols; ++j){
    //         //uint8_t val = myData[i + _stride + j];
    //         printf("%lf\t", soorah.at<double>(i, j));
    //     }
    // }
}

inline float theta(double r, double g, double b) {
    return acos((r - (g * 0.1) - (b * 0)) / sqrtf((r * r) + (g * g) + (b * b) - (r * g) - (r * b) - (g * b)));
}

inline float getHue(double r, double g, double b){
    return (b <= g) ? (theta(r, g, b) * 255.f / (2.f * M_PI)) : (((2.f * M_PI) - theta(r, g, b)) * 255.f / (2.f * M_PI));
}

inline float getLum(double r, double g, double b){
    return (0.210f * r) + (0.715f * g) + (0.072f * b);
}

inline float getSat(double r, double g, double b){
    return (max(r, max(g, b)) - min(r, min(g, b)));
}

//no need
void HSLColorThreshold(Mat &temp, bool color, uint8_t hueMax, uint8_t hueMin, uint8_t satMin){
    Mat output;
    CV_Assert(temp.channels() == 3);
    output.create(temp.size(), CV_8UC1);

    for(int i=0; i<temp.rows; ++i){
        const uchar *imgData = temp.ptr<uchar>(i);
        uchar *outData = output.ptr<uchar>(i);
        for(int j=0; j<temp.cols; ++j){
            uchar s = *imgData++;
            uchar l = *imgData++;
            uchar h = *imgData++;
            if(!color)
                *outData++ = ((h > hueMin && h < hueMax) && s > satMin) ? 255 : 0;
            else
                *outData++ = ((h > hueMax || h < hueMax) && s > satMin) ? 255 : 0;
        }
    }
    temp = output;
}

//no need
Mat toIHLS(Mat src){
    Mat temp;
    //
    CV_Assert(src.channels() == 3);

    temp = src.clone();
    for(auto it = temp.begin<Vec3b>(); it != temp.end<Vec3b>(); ++it){
        Vec3b bgr = (*it);
        (*it)[0] = getSat(double(bgr[0]), double(bgr[1]), double(bgr[2]));
        (*it)[1] = getLum(double(bgr[0]), double(bgr[1]), double(bgr[2]));
        (*it)[2] = getHue(double(bgr[0]), double(bgr[1]), double(bgr[2]));
    }
    HSLColorThreshold(temp, 0, 250, 15, 10);
    return temp;
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
        line( standard_hough, pt1, pt2, Scalar(0,0,255), 8, LINE_AA);
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

Mat sobelThreshold(Mat src, uint8_t min=100, uint8_t max=200, char type='x', int kernalSize=3){
    Mat srcBlured, srcGray, gradient;
    Mat gradientX, gradientY, absoluteGradientX, absoluteGradientY;
    //GaussianBlur to remove the noise
    GaussianBlur(src, srcBlured, Size(3,3), 0, 0, BORDER_DEFAULT);
    //Convert to grayscale
    cvtColor(srcBlured, srcGray, CV_BGR2GRAY);
    //#########################
    if(type == 'x'){
        Sobel(srcGray, gradientX, ddepth, 1, 0, kernalSize, scale, delta, BORDER_DEFAULT);
        convertScaleAbs(gradientX, absoluteGradientX);
        vector<vector<uint8_t> > img;
        img.resize(absoluteGradientX.rows);
        for(int i=0; i<absoluteGradientX.rows; ++i){
            vector<uint8_t> temp;
            temp.resize(absoluteGradientX.cols);
            uint8_t* p = absoluteGradientX.ptr<uint8_t>(i);
            for(int j=0; j<absoluteGradientX.cols; ++j){
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
    }
    else{
        Sobel(srcGray, gradientY, ddepth, 0, 1, kernalSize, scale, delta, BORDER_DEFAULT);
        convertScaleAbs(gradientY, absoluteGradientY);
        vector<vector<uint8_t> > img;
        img.resize(absoluteGradientY.rows);
        for(int i=0; i<absoluteGradientY.rows; ++i){
            vector<uint8_t> temp;
            temp.resize(absoluteGradientY.cols);
            uint8_t* p = absoluteGradientY.ptr<uint8_t>(i);
            for(int j=0; j<absoluteGradientY.cols; ++j){
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
    }
    //#########################
    // //Derivatives calculations in X and Y directions using Sobel
    //     //Gradient X
    // Sobel(srcGray, gradientX, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    //     //Gradient Y
    // Sobel(srcGray, gradientY, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    // //Scale, calculate absolute values and convert the result to 8-bit;
    // convertScaleAbs(gradientX, absoluteGradientX);
    // convertScaleAbs(gradientY, absoluteGradientY);
    // //Calculate the weighted sum of two arrays, (approximately!)
    // addWeighted(absoluteGradientX, 0.5, absoluteGradientY, 0.5, -75, gradient);
    
    // vector<vector<uint8_t> > img;
    // img.resize(gradient.rows);
    // for(int i=0; i<gradient.rows; ++i){
    //     vector<uint8_t> temp;
    //     temp.resize(gradient.cols);
    //     uint8_t* p = gradient.ptr<uint8_t>(i);
    //     for(int j=0; j<gradient.cols; ++j){
    //         //uint8_t val = myData[i + _stride + j];
    //         if(p[j] >= min && p[j] <= max)
    //             temp[j] = 255;
    //         else
    //             temp[j] = 0;
    //     }
    //     img[i] = temp;
    // }
    // Mat sobeled((int)img.size(), (int)img[0].size(), CV_8UC1);
    // for(int i=0; i<sobeled.rows; ++i){
    //     for(int j=0; j<sobeled.cols; ++j){
    //         sobeled.at<uint8_t>(i,j) = img[i][j];
    //     }
    // }
    // cvtColor(sobeled, sobeled, CV_GRAY2BGR);
    // return sobeled;
    
    // imshow(window_name, myImage);
    // waitKey(0);
}

// Mat colorThreshold(Mat src, uint8_t min = 100, uint8_t max = 200) {
//     cvtColor(src, src, CV_BGR2HLS);
//     Mat channels[3];
//     split(src, channels);
//     Mat srcHLS = channels[1];
//     vector<vector<uint8_t> > img;
//     img.resize(srcHLS.rows);
//     for(int i=0; i<srcHLS.rows; ++i){
//         vector<uint8_t> temp;
//         temp.resize(srcHLS.cols);
//         uint8_t* p = srcHLS.ptr<uint8_t>(i);
//         for(int j=0; j<srcHLS.cols; ++j){
//             if(p[j] > min && p[j] <= max)
//                 temp[j] = 255;
//             else
//                 temp[j] = 0;
//         }
//         img[i] = temp;
//     }
//     Mat resColor((int)img.size(), (int)img[0].size(), CV_8UC1);
//     for(int i=0; i<resColor.rows; ++i){
//         for(int j=0; j<resColor.cols; ++j){
//             resColor.at<uint8_t>(i,j) = img[i][j];
//         }
//     }
//     return resColor;
// }

Mat perspectiveTransformation(Mat input){
    Mat output;
    //Input Quadilateral or Image plane coordinates
    Point2f inputQuad[4];
    //Output Quadilateral or World plane coordinates
    Point2f outputQuad[4];
    //Lambda Matrix
    Mat lambda(2, 4, CV_32FC1);
    //Set the lambda matrix the same type and size as input
    lambda = Mat::zeros(input.rows, input.cols, input.type());
    //The 4 points that select quadilateral on the input , from top-left in clockwise order
    //These four pts are the sides of the rect box used as input 
    inputQuad[0] = Point2f(-30, -60);
    inputQuad[1] = Point2f(input.cols+50, -50);
    inputQuad[2] = Point2f(input.cols+100, input.rows+50);
    inputQuad[3] = Point2f(-50, input.rows+50);
    //The 4 points where the mapping is to be done , from top-left in clockwise order
    outputQuad[0] = Point2f(0, 0);
    outputQuad[1] = Point2f(input.cols-1, 0);
    outputQuad[2] = Point2f(input.cols-1, input.rows-1);
    outputQuad[3] = Point2f(0, input.rows-1);
    //Get the Perspective Transform Matrix
    lambda = getPerspectiveTransform(inputQuad, outputQuad);
    //Apply the Perspective Transform matrix to the input
    warpPerspective(input, output, lambda, output.size());
    // imshow("Input", input);
    // imshow("output", output);
    // waitKey(0);
    return output;
}

Mat directionThreshold(Mat src, uint8_t min = 0, uint8_t max = M_PI/2){
    Mat srcX, srcY, srcGray;
    //Convert to grayscale
    cvtColor(src, srcGray, CV_BGR2GRAY);
    //Derivatives calculations in X and Y directions using Sobel
        //Gradient X
    Sobel(srcGray, srcX, ddepth, 1, 0, 1, scale, delta, BORDER_DEFAULT);
        //Gradient Y
    Sobel(srcGray, srcY, ddepth, 0, 1, 1, scale, delta, BORDER_DEFAULT);
    //Scale, calculate absolute values and convert the result to 8-bit;
    convertScaleAbs(srcX, srcX);
    convertScaleAbs(srcY, srcY);
    //Calculate the direction of the gradient using arc tangent of y/x, expressed in radians
    //direction = atan(srcY, srcX) * 180/M_PI;
    Mat direction(srcX.rows, srcX.cols, CV_8UC1);
    for(int i=0; i<srcX.rows; ++i){
        for(int j=0; j<srcX.cols; ++j){
            direction.at<uint8_t>(i,j) = atan2(srcX.at<uint8_t>(i,j), srcY.at<uint8_t>(i,j)) * (180/M_PI);
        }
    }
    //Calculate threshold
    //Mat output = Mat::zeros(direction.rows, direction.cols, CV_8UC1, 0.0); // To fill with zeros on creation
    //Mat output(direction.rows, direction.cols, CV_8UC1, 0.0); // another method to fill zeros
    Mat output(direction.rows, direction.cols, CV_8UC1);
    for(int i=0; i<direction.rows; ++i){
        for(int j=0; j<direction.cols; ++j){
            if(direction.at<uint8_t>(i,j) >= min && direction.at<uint8_t>(i,j) <= max)
                output.at<uint8_t>(i,j) = 0;
            else
                output.at<uint8_t>(i,j) = 255;
        }
    }
    return output;
}

Mat magnitudeThreshold(Mat src, uint8_t min = 100, uint8_t max = 200){
    Mat srcX, srcY, srcGray;
    //Convert to grayscale
    cvtColor(src, srcGray, CV_BGR2GRAY);
    //Derivatives calculations in X and Y directions using sobel
        //Gradient X
    Sobel(srcGray, srcX, ddepth, 1, 0, 1, scale, delta, BORDER_DEFAULT);
        //Gradient Y
    Sobel(srcGray, srcY, ddepth, 0, 1, 1, scale, delta, BORDER_DEFAULT);
    //Scale, calculate absolute values and convert the result to 8-bit;
    convertScaleAbs(srcX, srcX);
    convertScaleAbs(srcY, srcY);
    //Calculate gradient magnitude
    double maximumScaleFactor = 0;
    //Mat gradMag(srcX.rows, srcX.cols, CV_8UC1);
    vector<vector<double> > gradMag;
    gradMag.resize(srcX.rows);
    for(int i=0; i<srcX.rows; ++i){
        vector<double> temp;
        temp.resize(srcX.cols);
        uint8_t* pX = srcX.ptr<uint8_t>(i);
        uint8_t* pY = srcY.ptr<uint8_t>(i);
        for(int j=0; j<srcX.cols; ++j){
            temp[j] = sqrt(pX[j]*pX[j] + pY[j]*pY[j]);
            if(temp[j] >= maximumScaleFactor)
                maximumScaleFactor = temp[j];
        }
        gradMag[i] = temp;
    }
    maximumScaleFactor /= 255;
    //Rescale to 8-bit
    Mat gradMag8(gradMag.size(), gradMag[0].size(), CV_8UC1);
    for(int i=0; i<gradMag8.rows; ++i){
        for(int j=0; j<gradMag8.cols; ++j){
            gradMag8.at<uint8_t>(i,j) = uint8_t(gradMag[i][j] / maximumScaleFactor);
        }
    }
    //Calculate threshold
    Mat output(gradMag8.rows, gradMag8.cols, CV_8UC1);
    for(int i=0; i<output.rows; ++i){
        for(int j=0; j<output.cols; ++j){
            if(gradMag8.at<uint8_t>(i,j) >= min && gradMag8.at<uint8_t>(i,j) <= max)
                output.at<uint8_t>(i,j) = 255;
            else
                output.at<uint8_t>(i,j) = 0;
        }
    }
    return output;
}

Mat combinedGradientThresholding(Mat sobeledX, Mat sobeledY, Mat dir, Mat mag){
    Mat output(sobeledX.rows, sobeledX.cols, CV_8UC1);
    for(int i=0; i<output.rows; ++i){
        for(int j=0; j<output.cols; ++j){
            if((sobeledX.at<uint8_t>(i,j) == 255 && sobeledY.at<uint8_t>(i,j) == 255) || (dir.at<uint8_t>(i,j) == 255 && mag.at<uint8_t>(i,j) == 255))
                output.at<uint8_t>(i,j) = 255;
            else
                output.at<uint8_t>(i,j) = 0;
        }
    }
    return output;
}

Mat HLS(Mat src, uint8_t min=160, uint8_t max=255){
    Mat hls;
    //Convert to HLS color space
    cvtColor(src, hls, CV_BGR2HLS);
    Mat output(src.rows, src.cols, CV_8UC1);
    //Apply threshold to the S channel
    //split(hls, channels);
    for(int i=0; i<hls.rows; ++i){
        for(int j=0; j<hls.cols; ++j){
            Vec3b intensity = hls.at<Vec3b>(i,j);
            if(intensity[2] > min && intensity[2] <= max)
                output.at<uint8_t>(i,j) = 255;
            else
                output.at<uint8_t>(i,j) = 0;
        }
    }
    //cout << output.type() << endl;
    return output;
}

Mat regionOfInterest(Mat src,const Point* v){
    Mat image = Mat::zeros(src.rows, src.cols, CV_8UC1);
    int ignore = 255;
    int num_points = 8;
    //fillPoly(image, v[0], &num_points, 1, Scalar(255,255,255), 4);
    Mat output(src.rows, src.cols, CV_8UC1);
    //cout << image.rows << " " << image.cols << endl;
    bitwise_and(src, image, output);
    return image;
    //return image;
}

Mat combinedColorThresholding(Mat hls, Mat combinedGradient){
    Mat output(hls.rows, hls.cols, CV_8UC1);
    for(int i=0; i<output.rows; ++i){
        for(int j=0; j<output.cols; ++j){
            if(hls.at<uint8_t>(i,j) == 255 || combinedGradient.at<uint8_t>(i,j) == 255)
                output.at<uint8_t>(i,j) = 255;
            else
                output.at<uint8_t>(i,j) = 0;
        }
    }
    Point leftBot(100, output.rows);
    Point rightBot(output.cols - 20, output.rows);
    Point innerLeftBot(310, output.rows);
    Point innerRightBot(1150, output.rows);
    Point apex1(610, 410);
    Point apex2(680, 410);
    Point innerApex1(700, 480);
    Point innerApex2(650, 480);

    // Point leftBot(100, output.rows);
    // Point rightBot(output.cols - 20, output.rows);
    // Point innerLeftBot(1000, output.rows);
    // Point innerRightBot(1150, output.rows);
    // Point apex1(1000, 1000);
    // Point apex2(1000, 1000);
    // Point innerApex1(1000, 1000);
    // Point innerApex2(1000, 1000);
    Point vertices[1][8];
    vertices[0][0] = leftBot;      vertices[0][1] = rightBot;
    vertices[0][2] = innerLeftBot; vertices[0][3] = innerRightBot;
    vertices[0][4] = apex1;        vertices[0][5] = apex2;
    vertices[0][6] = innerApex1;   vertices[0][7] = innerApex2;
    const Point* corner_list[1] = {vertices[0]};

    //####################
    Mat image = Mat::zeros(output.rows, output.cols, CV_8UC1);
    int ignore = 255;
    int num_points = 8;
    fillPoly(image, corner_list, &num_points, 1, Scalar(255,255,255), 8);
    //Mat output(src.rows, src.cols, CV_8UC1);
    //cout << image.rows << " " << image.cols << endl;
    bitwise_and(output, image, output);
    //####################

    //output = regionOfInterest(output, corner_list);
    return output;
}

int main(int argc, char** argv){
    //src = imread(argv[1]);
    //string path = "/Users/mohammedamarnah/Desktop/SDCProject/CBM64-SDC/Lane Detection/Lane Detection/Samples/";
    string path2 = argv[1];
    //VideoCapture cap(path2);
    cap.open(path2);
    namedWindow(window_name, WINDOW_AUTOSIZE);
    //############
    // int frames = (int) cap.get(CAP_PROP_FRAME_COUNT);
    // int tmpw   = (int) cap.get(CAP_PROP_FRAME_WIDTH);
    // int tmph   = (int) cap.get(CAP_PROP_FRAME_HEIGHT);
    // cout << "Video has " << frames << " Frames of dimensions(" << tmpw << ", " << tmph << ")." << endl;
    // createTrackbar("Position", window_name, &g_slider_position, frames, onTrackbarSlide);
    //############

    
    if(!cap.isOpened()){
        cout << "Error, not a valid video!" << endl;
        exit(1);
    }
    while(1){
        double time = clock();
        
        if(g_run != 0){
            cap >> src;
            if(src.empty())
                break;
            originalImage = src;
            //Cut the upper part of the image
            //Rect myROI(src.cols-(src.cols*0.9), src.rows/2, src.cols-(src.cols*0.1), src.rows/2);
            //src = src(myROI);
            
            if(!src.data)
                return -1;
            //originalImage = src;

            //IHLS
            //src = toIHLS(originalImage);

            
            //Apply Sobel Threshold
            //sobeledX = sobelThreshold(originalImage, 100, 200, 'x', 3);
            //sobeledY = sobelThreshold(originalImage, 100, 200, 'y', 3);
            
            //Apply Direction Threshold
            //dirThreshold = directionThreshold(originalImage, 0.65, 1.05);

            //Apply Magnitude Threshold
            //magThreshold = magnitudeThreshold(originalImage, 40, 255);

            //Apply Combined Thresholding
            //combined = combinedGradientThresholding(sobeledX, sobeledY, dirThreshold, magThreshold);

            //Apply HLS color Threshold
            //hlsed = HLS(originalImage, 100, 250);

            //Apply Combined Color Threshold
            //combinedColor = combinedColorThresholding(hlsed, combined);
            
            //Perspective Transformation
            //perspectiveTransformed = perspectiveTransformation(edges);
            
            //Apply Canny edge detector
            //Canny(combined, edges, 50, 200, 3);
            
            //Standard HoughTransform
            //standardHough(0, 0);

            //Test for CTF
            Mat output(src.rows, src.cols, CV_8UC1);
            for(int i=0; i<src.rows; ++i){
                for(int j=0; j<src.cols; ++j){
                    if(src.at<uint8_t>(i,j) >= 100)
                        output.at<uint8_t>(i,j) = 255;
                    else
                        output.at<uint8_t>(i,j) = 0;
                }
            }
            
            //flip(src, magThreshold, -1);
            
            // if (flag == 1)
            //     toShow = sobeled;
            // else if (flag == 2)
            //     toShow = colored;
            // else
            //     toShow = combined;

            // Compute the time elapsed in processing one frame
            double timeElapsed = clock() - time;
            timeElapsed = 1/(timeElapsed/CLOCKS_PER_SEC);
            putText(toShow, to_string(timeElapsed)+" FPS", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255,255,255), 1, CV_AA);
            
            //Show the result
            int current_pos = (int)cap.get(cv::CAP_PROP_POS_FRAMES);
            g_dontset = 1;
            setTrackbarPos("Position", window_name, current_pos);
            //pyrDown(originalImage, src);
            imshow(window_name, output);
            g_run -= 1;
        }

        char c = (char) waitKey(10);
        if(c == 's'){
            g_run = 1;
            cout << "Single step, run = " << g_run << endl;
        }
        if(c == 'r'){
            g_run = -1;
            cout << "Run mode, run = " << g_run << endl;
        }
        if(c == 27)
            break;
        // char c = (char)waitKey(25);
        // if(c==27)
        //     break;
        // else if (c == 32)
        //     flag = (flag + 1)%3;
    }
}
