//
// Created by user on 2018-10-31.
//

#ifndef DLIBDEMO_EYEBROW_H
#define DLIBDEMO_EYEBROW_H
#include <iostream>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include "dlib/opencv.h"
#include "opencv2/photo.hpp"

int eyebrow(cv::Mat src,std::vector<cv::Point> shape_points,cv::Mat tmp_left, cv::Mat tmp_right);
#endif //DLIBDEMO_EYEBROW_H
