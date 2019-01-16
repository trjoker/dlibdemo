//
// Created by 陶然 on 2018/10/31.
//

#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <iostream>
#include <time.h>
#include<opencv2\opencv.hpp>
using namespace std;

int  makeLip(cv::Mat picture,std::vector<cv::Point> shape_points);
