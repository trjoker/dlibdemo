//
// Created by lvfei on 2018/12/21.
//

#ifndef DLIBDEMO_BLUSH_H
#define DLIBDEMO_BLUSH_H
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>

cv::Point2f lineIntersection(cv::Point2f a, cv::Point2f b, float angle);
void matrix_calculate(cv::Point2f t1, cv::Point2f t2, cv::Point2f t3, cv::Mat &bmask, cv::Mat &src, float angle, bool is_left);
void blush(cv::Mat src, std::vector<cv::Point> points, cv::Mat bmask_left, cv::Mat bmask_right);

#endif //DLIBDEMO_BLUSH_H
