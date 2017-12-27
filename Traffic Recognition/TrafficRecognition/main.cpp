//
//  main.cpp
//  TrafficRecognition
//
//  Created by Mohammed Amarnah on 12/7/17.
//  Copyright © 2017 Mohammed Amarnah. All rights reserved.
//

#include "ColorProcessing.h"
#include "ImageProcessing.h"

using namespace std;

vector<vector<bool> > vis;
vector<vector<int> > v;
vector<vector<pair<int, int> > > sets;

cv::Scalar clr;
cv::Point a, b;

int n, m, maxX, minX, maxY, minY;

int dr[] = {0, 0, 1, -1, 1, -1, -1, 1};
int dc[] = {1, -1, 0, 0, 1, -1, 1, -1};

void bfs(int x, int y) {
    queue<pair<int, int>> q;
    vector<pair<int, int> > s;
    q.push({x, y});
    s.push_back({x,y});
    vis[x][y] = 1;
    
    while (!q.empty()) {
        int topx = q.front().first;
        int topy = q.front().second;
        q.pop();
        for (int i = 0; i < 8; i++) {
            int nr = dr[i] + topx;
            int nc = dc[i] + topy;
            if (nr < 0 || nr >= n || nc < 0 || nc >= m || vis[nr][nc] || v[nr][nc] == 0) continue;
            s.push_back({nr, nc});
            q.push({nr, nc});
            vis[nr][nc] = 1;
        }
    }
    
    sets.push_back(s);
    return;
}

void fillSets(cv::Mat &filledContours) {
    // Converts the image from Mat format to a vector<vector<int> > format.
    v.clear();
    vis.clear();
    v.resize(filledContours.rows, vector<int>(filledContours.cols));
    vis.resize(filledContours.rows, vector<bool>(filledContours.cols));
    for (int i = 0; i < filledContours.rows; i++) {
        for (int j = 0; j < filledContours.cols; j++) {
            v[i][j] = filledContours.at<uint8_t>(i, j);
        }
    }
    // Perform a Breadth-first search (BFS) and assigns every collection
    // of white pixels as a set of points.
    n = (int)v.size(), m = (int)v[0].size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (v[i][j] == 255 && !vis[i][j]) {
                bfs(i, j);
            }
        }
    }
}

void drawRectangles(cv::Mat &image) {
    // Iterates over the set of points found by the BFS algorithm
    // and draws a rectangle over the max point and the min point
    // in each set.
    clr[2] = 255;
    clr[1] = 0;
    clr[0] = 255;
    for (int i = 0; i < sets.size(); i++) {
        maxX = maxY = -1e4;
        minX = minY = 1e4;
        for (int j = 0; j < sets[i].size(); j++) {
            maxX = max(maxX, sets[i][j].first);
            maxY = max(maxY, sets[i][j].second);
            minX = min(minX, sets[i][j].first);
            minY = min(minY, sets[i][j].second);
        }
        a.x = minY;
        a.y = minX;
        b.x = maxY;
        b.y = maxX;
        if (abs(b.x-a.x) >= 32 && abs(b.y-a.y) >= 32)
            cv::rectangle(image, a, b, clr, 2);
    } sets.clear();
}

void runVideo(string path) {
    // Some definitions
    cv::Mat input;
    cv::VideoCapture cap(path);
    colorProcessing clrProc;
    imageProcessing imgProc;

    if (!cap.isOpened()) {
        cerr << "Error, not a valid video: using an image instead." << endl;
        return;
    }
    bool flag = 0;
    while (1) {
        double time = clock();
        cap >> input;
        
        // Convert to improved HLS color space
        cv::Mat IHLS = clrProc.BGR2IHLS(input);
        
        /* RED: 1, 240, 15, 50
         BLUE: 0, 163, 134, 39 */
        // Apply color threshold based on the saturation and the hue
        cv::Mat colorThresh = clrProc.colorThreshold(IHLS, 1, 240, 15, 150);
        
        // Finds the contours in the image and filters based on them
        cv::Mat filled = imgProc.filterContours(colorThresh);
        
        // Fills a vector with sets, every set defining a shape
        // using the BFS algorithm.
        fillSets(filled);
        
        // Finds the maximum point and the minimum point in each
        // set and draws a rectangle based on them.
        drawRectangles(input);
        
        // Assigns the output image
        cv::Mat out;
        if (flag) {
            out = filled;
        } else {
            out = input;
        }
        
        double timeElapsed = (clock() - time) / CLOCKS_PER_SEC;
        cv::putText(out, to_string(1/timeElapsed) + " FPS", cvPoint(30, 40), 3, 1, cvScalar(255));
        cv::imshow("output", out);

        char c = cv::waitKey(25);
        if (c == 27)
            break;
        else if (c == 32) {
            flag ^= 1;
        }
    }
    exit(0);
}

int main() {
    // Start counting, bruh!
    double time = clock();
    
    colorProcessing clrProc;
    imageProcessing imgProc;
    
    string path = "/Users/mohammedamarnah/Desktop/SDCProject/CBM64-SDC/Traffic Recognition/data/";
    string file = "video2.mp4";
    
    // You can run the video file by only passing the path parameter.
    runVideo(path + file);
    cv::Mat input = cv::imread(path + file);
    
    // Convert to improved HLS color space.
    cv::Mat IHLS = clrProc.BGR2IHLS(input);
    
    /* RED: 1, 240, 15, 50
     BLUE: 0, 163, 134, 39 */
    // Apply color threshold based on the saturation and the hue.
    cv::Mat colorThresh = clrProc.colorThreshold(IHLS, 1, 240, 15, 150);
    
    // Finds the contours in the image and filters based on them.
    cv::Mat filled = imgProc.filterContours(colorThresh);
    
    // Fills a vector with sets, every set defining a shape
    // using the BFS algorithm.
    fillSets(filled);
    
    // Finds the maximum point and the minimum point in each
    // set and draws a rectangle based on them.
    drawRectangles(input);
    
    // Finds the convex hull contours and filters based on them
    // vector<vector<cv::Point>> contours;
    // imgProc.contourExtraction(filteredContours, contours);
    
    // Assigns the output image
    cv::Mat out = input;
    
    double timeElapsed = (clock() - time) / CLOCKS_PER_SEC;
    cv::putText(out, to_string(1/timeElapsed) + " FPS", cvPoint(30, 40), 2, 1, cvScalar(0));
    cv::imshow("output", out);
    cv::waitKey(0);
}
