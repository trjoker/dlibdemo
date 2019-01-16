//
// Created by 陶然 on 2018/12/3.
//

#ifndef DLIBDEMO_EYELASH_H
#define DLIBDEMO_EYELASH_H

#endif //DLIBDEMO_EYELASH_H

#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <iostream>
#include <time.h>
#include <stdint.h>
#include "ImageWarp.h"
#include <opencv2/imgproc.hpp>
#include<opencv2\opencv.hpp>
#include <opencv2/core.hpp>

using namespace std;


void eyelash(cv::Mat &dst, cv::Mat lash, std::vector<cv::Point> shape_points);




