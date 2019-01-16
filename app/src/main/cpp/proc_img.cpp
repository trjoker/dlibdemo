#include<opencv2\opencv.hpp>
#include "dlib_proc.h"
#include <dlib/opencv.h>

using namespace cv;
//using namespace dlib;
using namespace std;

int paint_points(cv::Mat img, std::vector<full_object_detection> shape_points) {
    for (int i = 0; i < shape_points.size(); i++) {
        for (int j = 0; j < shape_points[i].num_parts(); j++)
            circle(img, cv::Point(shape_points[i].part(j)(0), shape_points[i].part(j)(1)), 2,
                   cv::Scalar(0, 0, 255), -1);
    }
    return 0;
}

int proc_img(Mat picture) {

    std::vector<full_object_detection> shape_points;
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    dlib::array2d<dlib::bgr_pixel> dlib_image;
    cv_image<dlib::bgr_pixel> dlib_img(picture);
    shape_points = face_detect_mkup(dlib_img);
    paint_points(picture, shape_points);

    return 0;
}

