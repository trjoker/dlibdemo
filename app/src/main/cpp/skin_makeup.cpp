#include<opencv2\opencv.hpp>
#include "skin_makeup.h"
using namespace cv;

Mat proc(Mat image,float p) {
    Mat dst = Mat();

    int value1 = 3, value2 = 1;
    int dx = value1 * 5;
    float fc = value1 * 12.5;

    Mat temp1 = Mat(), temp2 =  Mat(), temp3 =  Mat(), temp4 = Mat();

    bilateralFilter(image, temp1, dx, fc, fc);
    Mat temp22 = Mat();
    subtract(temp1, image, temp22);
    add(temp22, Scalar(128, 128, 128, 128), temp2);
    GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

    Mat temp44 = Mat();
    temp3.convertTo(temp44, temp3.type(), 2, -255);
    add(image, temp44, temp4);
    addWeighted(image, p, temp4, 1 - p, 0.0, dst);

    //add(dst, Scalar(10, 10, 20), dst);
    return dst;

}
int smooth_skin(Mat srcImage) {
    Mat dst;
    float p = 0.2;
    dst = proc(srcImage,p);
    dst.copyTo(srcImage);
    return 0;
}

