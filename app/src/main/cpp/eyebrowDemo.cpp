//eyebrow makeup demo

#include "eyebrow.h"

using namespace std;
using namespace cv;

int eyebrow(cv::Mat src,std::vector<cv::Point> shape_points, cv::Mat tmp_left, cv::Mat tmp_right)
{
	//string path = argv[1];
	//string path = "/storage/emulated/0/imageTest.jpg";

	//Mat src;
	//src = imread(path);
	//vector<dlib::full_object_detection> shape_points;
	//eyebrow points detect
	//shape_points = Detect(src);
	vector<cv::Point> eyepoint(10);
	if (!shape_points.empty())
	{
		for (int i = 17; i <= 26; i++)
		{
			eyepoint[i - 17].x = shape_points[i].x;
			eyepoint[i - 17].y = shape_points[i].y;
		}
	}
	//eyebrow temple
	//Mat tmp_left; //= imread("/storage/emulated/0/eyebrow_zuo.png");
	//Mat tmp_right; //= imread("/storage/emulated/0/eyebrow_you.png");
	Mat left_dst;
	Mat right_dst;

	//eyebrow scale
	float scale_left_y;
	float scale_left_x = 1.0;
	scale_left_y = float(abs(eyepoint[4].x - eyepoint[0].x )) / (tmp_left.cols);
	
	//cout << "scale_left_x" << scale_left_x;
	//cout << "tmp_left.cols" << tmp_left.cols;
	//cout << "eyepoint" << eyepoint;
	//cout << "eyepoint[4].x" << eyepoint[4].x;
	resize(tmp_left, left_dst, Size(0, 0), scale_left_x, scale_left_y, INTER_LINEAR);

    //right eye brow temple scale
    float scale_right_x = 1.0;
    float scale_right_y;
    scale_right_y = float(abs(eyepoint[9].x - eyepoint[5].x)) / (tmp_right.cols);
    resize(tmp_right, right_dst, Size(0, 0), scale_right_x, scale_right_y, INTER_LINEAR);


    // Create an all white mask
    Mat left_dst_mask = 255 * Mat::ones(left_dst.rows, left_dst.cols, left_dst.depth());
    Mat right_dst_mask = 255 * Mat::ones(right_dst.rows, right_dst.cols, right_dst.depth());
    // The location of the center of the src in the dst
    //Point center(dst.cols / 2, dst.rows / 2);
    Point center_left(eyepoint[2].x, eyepoint[2].y + 4 * (eyepoint[4].y - eyepoint[2].y) / 5);
    Point center_right(eyepoint[7].x, eyepoint[7].y + 4 * (eyepoint[4].y - eyepoint[2].y) / 5);
    // Seamlessly clone src into dst and put the results in output
    Mat normal_clone;
    Mat mixed_clone;
    Mat monochrome_clone;

    seamlessClone(left_dst, src, left_dst_mask, center_left, mixed_clone,  MIXED_CLONE);
    seamlessClone(right_dst, mixed_clone, right_dst_mask, center_right, src, MIXED_CLONE);




    /*

	//left eyebrow landmark
	int x_offset = eyepoint[0].x;
	int y_offset = eyepoint[0].y;
	int y1 = y_offset - left_dst.rows;
	int y2 = y_offset;
	int x1 = x_offset;
	int x2 = x_offset + left_dst.cols;

	//image ROI
	cv::Mat imageROI_left = src(cv::Rect(eyepoint[0].x, eyepoint[0].y  - 28, left_dst.cols, left_dst.rows));

	//setting parameters
	double alpha = 1.0;
	double beta = 1 - alpha -0.88;
	double gamma = 0.0;

	//add image
	addWeighted(imageROI_left, alpha, left_dst, beta, gamma, imageROI_left);

	//right eye brow temple scale
	float scale_right_x;
	scale_right_x = float(abs(eyepoint[9].x - eyepoint[5].x)) / (tmp_right.cols);
	cv::resize(tmp_right, right_dst, cv::Size(0, 0), scale_right_x, scale_right_x, cv::INTER_LINEAR);
	//right eyebrow landmark
	int r_x_offset = eyepoint[5].x;
	int r_y_offset = eyepoint[5].y;
	int r_y1 = r_y_offset - right_dst.rows;
	int r_y2 = r_y_offset;
	int r_x1 = r_x_offset;
	int r_x2 = r_x_offset + right_dst.cols;

	//eyebrow ROI
	cv::Mat imageROI_right = src(cv::Rect(eyepoint[5].x, eyepoint[5].y - 28 , right_dst.cols, right_dst.rows));
	//ROI
	addWeighted(imageROI_right, alpha, right_dst, beta, gamma, imageROI_right);
	//show image

	//imshow("image_show", src);
	//waitKey(0);

    */
    return 0;
}
