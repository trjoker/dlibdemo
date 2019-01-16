//eyeliner makeup demo

#include "eyeliner.h"
using namespace std;
using namespace cv;

//关键点检测
bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
    //Number of key points
    int N = key_point.size();

    //构造矩阵X
    cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            for (int k = 0; k < N; k++)
            {
                X.at<double>(i, j) = X.at<double>(i, j) +
                                     std::pow(key_point[k].x, i + j);
            }
        }
    }

    //构造矩阵Y
    cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int k = 0; k < N; k++)
        {
            Y.at<double>(i, 0) = Y.at<double>(i, 0) +
                                 std::pow(key_point[k].x, i) * key_point[k].y;
        }
    }

    A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    //求解矩阵A
    cv::solve(X, Y, A, cv::DECOMP_LU);
    return true;
}


int eyeliner_makeup(cv::Mat dst,std::vector<cv::Point> shape_points)
{

    //提取眼线关键点

    std::vector<cv::Point> eyepoint(20);
    if (!shape_points.empty())//左眼线和右眼线坐标
    {
        for (int i = 36; i <= 47; i++)
        {
            eyepoint[i - 36].x = shape_points[i].x;
            eyepoint[i - 36].y = shape_points[i].y;
            //std::cout << "eyepoint[i-47]" << eyepoint[i-36]<< std::endl;
        }
    }

    //左眼上关键点精度调优eyepoint[0],eyepoint[1],eyepoint[2],eyepoint[3]
    eyepoint[0].y = eyepoint[0].y - 2;
    eyepoint[1].y = eyepoint[1].y - 1;
    eyepoint[2].y = eyepoint[2].y - 1;

    eyepoint[0].x = eyepoint[0].x - 5;
    eyepoint[1].x = eyepoint[1].x - 1;
    eyepoint[2].x = eyepoint[2].x - 1;

    //左眼下关键点精度调优eyepoint[0],eyepoint[5],eyepoint[4],eyepoint[3]
    eyepoint[5].y = eyepoint[5].y + 1;
    eyepoint[4].y = eyepoint[4].y + 1;

    eyepoint[5].x = eyepoint[5].x - 1;
    eyepoint[4].x = eyepoint[4].x - 1;

    //右眼上关键点精度调优eyepoint[6],eyepoint[7],eyepoint[8],eyepoint[9]
    eyepoint[9].y = eyepoint[9].y - 2;
    eyepoint[8].y = eyepoint[8].y - 1;
    eyepoint[7].y = eyepoint[7].y - 1;

    eyepoint[9].x = eyepoint[9].x + 5;
    eyepoint[8].x = eyepoint[8].x + 1;
    eyepoint[7].x = eyepoint[7].x + 1;

    //左眼下关键点精度调优eyepoint[6],eyepoint[11],eyepoint[10],eyepoint[9]
    eyepoint[11].y = eyepoint[11].y + 1;
    eyepoint[10].y = eyepoint[10].y + 1;

    eyepoint[11].x = eyepoint[11].x + 1;
    eyepoint[10].x = eyepoint[10].x + 1;



    //输入拟合点  左眼上关键点eyepoint[0],eyepoint[1],eyepoint[2],eyepoint[3]
    std::vector<cv::Point> points_up_left;
    points_up_left.push_back(eyepoint[0]);
    points_up_left.push_back(eyepoint[1]);
    points_up_left.push_back(eyepoint[2]);
    points_up_left.push_back(eyepoint[3]);

    //输入拟合点  左眼下关键点eyepoint[0],eyepoint[5],eyepoint[4],eyepoint[3]
    std::vector<cv::Point> points_down_left;
    points_down_left.push_back(eyepoint[0]);
    points_down_left.push_back(eyepoint[5]);
    points_down_left.push_back(eyepoint[4]);
    points_down_left.push_back(eyepoint[3]);


    //输入拟合点  右眼上关键点eyepoint[6],eyepoint[7],eyepoint[8],eyepoint[9]
    std::vector<cv::Point> points_up_right;
    points_up_right.push_back(eyepoint[6]);
    points_up_right.push_back(eyepoint[7]);
    points_up_right.push_back(eyepoint[8]);
    points_up_right.push_back(eyepoint[9]);

    //输入拟合点  右眼下关键点eyepoint[6],eyepoint[11],eyepoint[10],eyepoint[9]
    std::vector<cv::Point> points_down_right;
    points_down_right.push_back(eyepoint[6]);
    points_down_right.push_back(eyepoint[11]);
    points_down_right.push_back(eyepoint[10]);
    points_down_right.push_back(eyepoint[9]);


    cv::Mat A;//左眼上矩阵
    cv::Mat A2;//左眼下矩阵
    cv::Mat A3;//右眼上矩阵
    cv::Mat A4;//右眼下矩阵

    //1,2,3,4-->左眼上，左眼下，右眼上，右眼下
    polynomial_curve_fit(points_up_left, 3, A);
    polynomial_curve_fit(points_down_left, 3, A2);
    polynomial_curve_fit(points_up_right, 3, A3);
    polynomial_curve_fit(points_down_right, 3, A4);

    //1,2,3,4-->左眼上，左眼下，右眼上，右眼下
    std::vector<cv::Point> points_fitted;
    std::vector<cv::Point> points_fitted2;
    std::vector<cv::Point> points_fitted3;
    std::vector<cv::Point> points_fitted4;

    for (int x = points_up_left[0].x; x < points_up_left[3].x; x++)
    {
        double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
                   A.at<double>(2, 0)*std::pow(x, 2) + A.at<double>(3, 0)*std::pow(x, 3);

        points_fitted.push_back(cv::Point(x, y));
    }


    for (int x = points_down_left[0].x; x < points_down_left[3].x; x++)
    {
        double y = A2.at<double>(0, 0) + A2.at<double>(1, 0) * x +
                   A2.at<double>(2, 0)*std::pow(x, 2) + A2.at<double>(3, 0)*std::pow(x, 3);

        points_fitted2.push_back(cv::Point(x, y));
    }


    for (int x = points_up_right[0].x; x < points_up_right[3].x; x++)
    {
        double y = A3.at<double>(0, 0) + A3.at<double>(1, 0) * x +
                   A3.at<double>(2, 0)*std::pow(x, 2) + A3.at<double>(3, 0)*std::pow(x, 3);

        points_fitted3.push_back(cv::Point(x, y));
    }


    for (int x = points_down_right[0].x; x < points_down_right[3].x; x++)
    {
        double y = A4.at<double>(0, 0) + A4.at<double>(1, 0) * x +
                   A4.at<double>(2, 0)*std::pow(x, 2) + A4.at<double>(3, 0)*std::pow(x, 3);

        points_fitted4.push_back(cv::Point(x, y));
    }

    cv::polylines(dst, points_fitted, false, cv::Scalar(30, 30, 30), 2, 8, 0);
    cv::polylines(dst, points_fitted2, false, cv::Scalar(42, 42, 42), 1, 8, 0);
    cv::polylines(dst, points_fitted3, false, cv::Scalar(30, 30, 30), 2, 8, 0);
    cv::polylines(dst, points_fitted4, false, cv::Scalar(42, 42, 42), 1, 8, 0);

    //cv::imshow("image", dst);
    //cv::waitKey(0);

    return 0;

}







