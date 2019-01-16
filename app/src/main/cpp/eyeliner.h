
#include <iostream>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include "dlib/opencv.h"
#include "opencv2/photo.hpp"

int  eyeliner_makeup(cv::Mat dst,std::vector<cv::Point> shape_points);