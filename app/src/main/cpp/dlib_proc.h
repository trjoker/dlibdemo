#ifndef DLIB_PROC_H
#define DLIB_PROC_H
#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <iostream>
#include <time.h>
#include<opencv2\opencv.hpp>
using namespace dlib;

template <typename image_type>
std::vector<full_object_detection> face_detect_mkup(image_type &img) {
	frontal_face_detector detector = get_frontal_face_detector();

	shape_predictor sp;
	deserialize("/storage/emulated/0/shape_predictor_68_face_landmarks.dat") >> sp;
	std::vector<rectangle> dets = detector(img);

	std::vector<full_object_detection> shapes;

	for (unsigned long j = 0; j < dets.size(); ++j)
	{
		full_object_detection shape = sp(img, dets[j]);
		shapes.push_back(shape);
	}

	return shapes;
};
int proc_img(cv::Mat picture);
#endif // !DLIB_PROC_H
//int  makeLip(cv::Mat picture,std::vector<cv::Point> shape_points);