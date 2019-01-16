//
//嘴唇化妆
// Created by 陶然 on 2018/10/29.
//
#include<opencv2\opencv.hpp>
#include "lip_makeup.h"
#include <dlib/opencv.h>
#include <android/log.h>
#include <ctime>
#include <stdio.h>
#include <time.h>

using namespace cv;
using namespace std;

//嘴唇区域放大边界，以防点位不准
int boundary = 30;

//三次贝塞尔曲线
float bezier3funcX(float uu, Point *controlP) {
    float part0 = controlP[0].x * uu * uu * uu;
    float part1 = 3 * controlP[1].x * uu * uu * (1 - uu);
    float part2 = 3 * controlP[2].x * uu * (1 - uu) * (1 - uu);
    float part3 = controlP[3].x * (1 - uu) * (1 - uu) * (1 - uu);
    return part0 + part1 + part2 + part3;
}

float bezier3funcY(float uu, Point *controlP) {
    float part0 = controlP[0].y * uu * uu * uu;
    float part1 = 3 * controlP[1].y * uu * uu * (1 - uu);
    float part2 = 3 * controlP[2].y * uu * (1 - uu) * (1 - uu);
    float part3 = controlP[3].y * (1 - uu) * (1 - uu) * (1 - uu);
    return part0 + part1 + part2 + part3;
}


//曲线拟合
std::vector<Point> createCurve(std::vector<Point> originPoint) {
    std::vector<Point> curvePoint;
    int originCount = originPoint.size();
    //控制点收缩系数 ，经调试0.6较好，CvPoint是opencv的，可自行定义结构体(x,y)
    float scale = 0.6;
    //CvPoint midpoints[12];
    std::vector<Point> midpoints;
    //生成中点
    for (int i = 0; i < originCount; i++) {
        int nexti = (i + 1) % originCount;
        //midpoints[i].x = (originPoint[i].x + originPoint[nexti].x) / 2.0;
        //midpoints[i].y = (originPoint[i].y + originPoint[nexti].y) / 2.0;
        midpoints.push_back(Point((originPoint[i].x + originPoint[nexti].x) / 2.0,
                                  (originPoint[i].y + originPoint[nexti].y) / 2.0));
    }

    //平移中点
    std::vector<Point> extrapoints(2 * originCount, Point(0, 0));
    for (int i = 0; i < originCount; i++) {
        int nexti = (i + 1) % originCount;
        int backi = (i + originCount - 1) % originCount;
        Point midinmid;
        midinmid.x = (midpoints[i].x + midpoints[backi].x) / 2.0;
        midinmid.y = (midpoints[i].y + midpoints[backi].y) / 2.0;
        int offsetx = originPoint[i].x - midinmid.x;
        int offsety = originPoint[i].y - midinmid.y;
        int extraindex = 2 * i;
        extrapoints[extraindex].x = midpoints[backi].x + offsetx;
        extrapoints[extraindex].y = midpoints[backi].y + offsety;
        //朝 originPoint[i]方向收缩
        int addx = (extrapoints[extraindex].x - originPoint[i].x) * scale;
        int addy = (extrapoints[extraindex].y - originPoint[i].y) * scale;
        extrapoints[extraindex].x = originPoint[i].x + addx;
        extrapoints[extraindex].y = originPoint[i].y + addy;

        int extranexti = (extraindex + 1) % (2 * originCount);
        extrapoints[extranexti].x = midpoints[i].x + offsetx;
        extrapoints[extranexti].y = midpoints[i].y + offsety;
        //朝 originPoint[i]方向收缩
        addx = (extrapoints[extranexti].x - originPoint[i].x) * scale;
        addy = (extrapoints[extranexti].y - originPoint[i].y) * scale;
        extrapoints[extranexti].x = originPoint[i].x + addx;
        extrapoints[extranexti].y = originPoint[i].y + addy;
    }

    Point controlPoint[4];
    //生成4控制点，产生贝塞尔曲线
    for (int i = 0; i < originCount; i++) {
        controlPoint[0] = originPoint[i];
        int extraindex = 2 * i;
        controlPoint[1] = extrapoints[extraindex + 1];
        int extranexti = (extraindex + 2) % (2 * originCount);
        controlPoint[2] = extrapoints[extranexti];
        int nexti = (i + 1) % originCount;
        controlPoint[3] = originPoint[nexti];
        float u = 1;
        while (u >= 0) {
            int px = bezier3funcX(u, controlPoint);
            int py = bezier3funcY(u, controlPoint);
            //u的步长决定曲线的疏密
            // u -= 0.005;
            u -= 0.05;
            Point tempP = Point(px, py);
            //存入曲线点
            curvePoint.push_back(tempP);
        }
    }
    return curvePoint;
}


//alpha blend
int cvAdd4cMat_q(cv::Mat &dst, cv::Mat &scr, double scale) {
    if (dst.channels() != 3 || scr.channels() != 4) {
        return true;
    }
    if (scale < 0.01)
        return false;
    std::vector<cv::Mat> scr_channels;
    std::vector<cv::Mat> dstt_channels;
    split(scr, scr_channels);
    split(dst, dstt_channels);

    //bgra   3 是透明度通道
    if (scale < 1) {
        scr_channels[3] *= scale;
        scale = 1;
    }


    for (int i = 0; i < 3; i++) {
        dstt_channels[i] = dstt_channels[i].mul(255.0 / scale - scr_channels[3], scale / 255.0);
        dstt_channels[i] += scr_channels[i].mul(scr_channels[3], scale / 255.0);
    }
    merge(dstt_channels, dst);
    return true;

}

//alpha blend
int cvAdd4cMat_q2(cv::Mat &dst, cv::Mat &scr, double scale) {


    addWeighted(dst, 0.5, scr, 0.5, 0, dst);
    return true;

}


//羽化功能
int
feather(Mat src, std::vector<Point> outPoints, int max_distance, int leftOut = 0, int topOut = 0) {
    //拿到嘴唇点位
    //vector<Point> lip_pts;
    double l_sum = 0;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            Vec4b a = src.at<Vec4b>(i, j);
            //printf("a, %d %d %d %d \n", a[0], a[1], a[2], a[3]);
            //判断嘴唇

            if (src.at<Vec4b>(i, j)[3] != 0) {
                //printf("src.at<Vec4b>(i, j)[3] %d   i %d  j %d  \n", src.at<Vec4b>(i, j)[3],j,i);
                ////lip_pts.push_back(Point(i, j));
                //对边界附近点位的进行模糊处理，越靠近边界的点越模糊
                double distance = pointPolygonTest(outPoints,
                                                   Point(j + leftOut - boundary,
                                                         i + topOut - boundary), 1);
                //printf("i, j %d %d \n", i,j);
                //printf("distance %lf \n", distance);
                if (distance < 0) distance = -distance;
                if (distance <= max_distance) {
//                    src.at<Vec4b>(i, j)[0] = 0;
//                    src.at<Vec4b>(i, j)[1] = 0;
//                    src.at<Vec4b>(i, j)[2] = 0;
                    src.at<Vec4b>(i, j)[3] = src.at<Vec4b>(i, j)[3] * (distance / max_distance);
                    //printf("src.at<Vec4b>(i, j)[3] after %d \n", src.at<Vec4b>(i, j)[3]);
                }
            }
        }
    }
    return 0;

}

long getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}


//嘴唇化妆
int makeLip(Mat originalMat, std::vector<cv::Point> shape_points) {
    long start0 = getCurrentTime();
    long start4 = getCurrentTime();
    long start5 = getCurrentTime();
    std::vector<cv::Point> pointsOut;
    std::vector<cv::Point> pointsIn;

    if (!shape_points.empty()) {

        for (int i = 0; i < 68; i++) {
            /*	circle(originalMat, Point(landmarks[0][i].x, landmarks[0][i].y),
                    3, Scalar(0, 0, 0), -1);*/
            //51 是上嘴唇中间点
            //if (i >= 48 && i < 60&&i!=51) {
            if (i >= 48 && i < 60) {
                pointsOut.push_back(shape_points[i]);
                /*circle(originalMat, Point(landmarks[0][i].x, landmarks[0][i].y),
                    1, Scalar(0, 255, 0), -1);*/
            }
            if (i >= 60 && i < 68) {
                pointsIn.push_back(shape_points[i]);
                /*	circle(originalMat, Point(landmarks[0][i].x, landmarks[0][i].y),
                        3, Scalar(0, 0, 0), -1);*/
            }

        }
    }
    long end5 = getCurrentTime();
    __android_log_print(ANDROID_LOG_INFO, "taoran", "点位初始化耗时 : %ld ms", (end5 - start5));
    long start6 = getCurrentTime();
    std::vector<std::vector<Point>> contours;
    contours.push_back(pointsOut);
    contours.push_back(pointsIn);
    std::vector<std::vector<Point>> hull(contours.size());



    //将嘴唇区域的矩形裁剪出来 48~62的点位中寻找最左、最右、最上、最下坐标
    int leftOut = (int) pointsOut[0].x;
    int rightOut = (int) pointsOut[6].x;
    int topOut = (int) pointsOut[2].y;
    int bottomOut = (int) pointsOut[9].y;
    for (int i = 0; i < contours.size(); i++) {
        for (int j = 0; j < contours[i].size(); j++) {

            int x = (int) contours[i][j].x;
            int y = (int) contours[i][j].y;
            if (leftOut > x) {
                leftOut = x;
            }
            if (rightOut < x) {
                rightOut = x;
            }
            if (topOut > y) {
                topOut = y;
            }
            if (bottomOut < y) {
                bottomOut = y;
            }
        }
    }

    int heightIn = pointsIn[6].y - pointsIn[2].y;
    long end6 = getCurrentTime();
    __android_log_print(ANDROID_LOG_INFO, "taoran", "寻找嘴唇区域耗时 : %ld ms", (end6 - start6));
    long start7 = getCurrentTime();
    Mat lip_map(originalMat.size(), CV_8UC4, Scalar(255, 255, 255, 0));

    long end7 = getCurrentTime();
    __android_log_print(ANDROID_LOG_INFO, "taoran", "创建合成图片耗时 : %ld ms", (end7 - start7));

    long end4 = getCurrentTime();
    __android_log_print(ANDROID_LOG_INFO, "taoran", "其他耗时 : %ld ms", (end4 - start4));
    long start = getCurrentTime();
    //曲线拟合
    //对外嘴唇点位进行拟合
    std::vector<Point> fitting_out_points;
    fitting_out_points = createCurve(contours[0]);
    contours[0] = fitting_out_points;
    drawContours(lip_map, contours, 0, Scalar(81, 79, 226, 255), -1, 16, std::vector<Vec4i>(), 0,
                 Point());
    //判断内嘴唇区域是否有嘴唇
    //当内嘴唇上下间距小于嘴唇上下间距的1/8时，定为没有张嘴，不填充内嘴唇轮廓。
    if ((bottomOut - topOut) < 8 * heightIn) {
        std::vector<Point> fitting_int_points;
        fitting_int_points = createCurve(contours[1]);
        contours[1] = fitting_int_points;
        drawContours(lip_map, contours, 1, Scalar(0, 0, 0, 0), -1, 16, std::vector<Vec4i>(), 0,
                     Point());
    }
    long end = getCurrentTime();
    __android_log_print(ANDROID_LOG_INFO, "taoran", "拟合耗时 : %ld ms", (end - start));
    long start2 = getCurrentTime();
    Rect rect = Rect(leftOut - boundary, topOut - boundary, rightOut - leftOut + boundary * 2,
                     bottomOut - topOut + boundary * 2);
    Mat roi_img = lip_map(rect);

    feather(roi_img, pointsOut, (bottomOut - topOut) / 8, leftOut, topOut);
    long end2 = getCurrentTime();
    __android_log_print(ANDROID_LOG_INFO, "taoran", "羽化耗时 : %ld ms", (end2 - start2));

    long start3 = getCurrentTime();
    Mat roi_img2 = originalMat(rect);
    cvAdd4cMat_q(roi_img2, roi_img, 0.5);
    long end3 = getCurrentTime();
    __android_log_print(ANDROID_LOG_INFO, "taoran", "合成耗时 : %ld ms", (end3 - start3));

    long end0 = getCurrentTime();
    __android_log_print(ANDROID_LOG_INFO, "taoran", "makeLip  耗时 : %ld ms", (end0 - start0));
    return 0;
}