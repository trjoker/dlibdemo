//
// Created by lvfei on 2018/12/21.
//

#include "blush.h"
using namespace std;
using namespace cv;

//a为 angle斜率上的，b为垂直angle的
cv::Point2f lineIntersection(cv::Point2f a, cv::Point2f b,float angle)
{
    cv::Point2f intersec;
    if (abs(abs(angle) -1.57)<0.01)//区域不会超过180度,弧度值3.14
    {
        intersec.x = a.x;
        intersec.y = b.y;
        return intersec;
    }
    float k = std::tan(angle);
    intersec.x = (b.y - a.y + k * a.x + b.x / k) / (k + 1 / k);
    intersec.y = (intersec.x - a.x)*k + a.y;
    return intersec;
}

void matrix_calculate(cv::Point2f t1, cv::Point2f t2, cv::Point2f t3, cv::Mat &bmask, cv::Mat &src, float angle, bool is_left)
{
    Size2f siz(sqrt((t2.x - t3.x)*(t2.x - t3.x) + (t2.y - t3.y)*(t2.y - t3.y)), sqrt((t1.x - t3.x)*(t1.x - t3.x) + (t1.y - t3.y)*(t1.y - t3.y)));
    cv::Point2f center = (t1 + t2) / 2;
    angle = (-angle - 3.1415926 / 2) * 180 / 3.1415926;//弧度转角度，注意弧度范围
    cv::RotatedRect rotated_rect(center, siz, angle);//得到带倾斜角的面部矩阵
    if(is_left)
        bmask = bmask(Range(0, 450), Range(0, 390));
    else
        bmask = bmask(Range(0, 450), Range(5, 395));
    //模板缩放（对角比例和非对角比例缩放），旋转，对齐,zuo
    Size2i source_size(bmask.cols, bmask.rows);
    Size2f target_size = rotated_rect.size;
    cv::Vec2f scale;//尺寸差异
    scale[0] = target_size.width / source_size.width;
    scale[1] = target_size.height / source_size.height;
    //缩放
    cv::resize(bmask, bmask, Size(int(source_size.width*scale[0]), int(source_size.height*scale[1])), 0.0, 0.0, CV_INTER_AREA);
    Point2f source_center((source_size.width*scale[0] - 1) / 2.0F, (source_size.height*scale[1] - 1) / 2.0F);
    //模板问题，需加入平移变化（可因模板位置变化相应调整），平移量根据图像长宽比例变化
    //定义平移矩阵。 模板本身相对自身有效区域太小，尝试缩放前进行下方裁剪
    cv::Mat t_mat = cv::Mat::zeros(2, 3, CV_32FC1);
    t_mat.at<float>(0, 0) = 1;
    if(is_left)
        t_mat.at<float>(0, 2) = -bmask.cols / 100.0; //水平平移量 ,左模板忘左，右模板向右
    else
        t_mat.at<float>(0, 2) = bmask.cols / 100.0;
    t_mat.at<float>(1, 1) = 1;
    t_mat.at<float>(1, 2) = bmask.rows / 9.0; //竖直平移量
    cv::warpAffine(bmask, bmask, t_mat, cv::Size(bmask.cols, bmask.rows), cv::INTER_LINEAR, 0, cv::Scalar(0, 0, 0, 0));
    //旋转，有角度限制，此处可以修改
    cv::Mat M = cv::getRotationMatrix2D(source_center, angle, 1.0);
    cv::warpAffine(bmask, bmask, M, cv::Size(bmask.cols, bmask.rows), cv::INTER_LINEAR, 0, cv::Scalar(0, 0, 0, 0));
    //坐标对齐与融合
    Point2i origin = rotated_rect.center - source_center;
    int color[3] = { 255,101,161 };
    float tmp = 0;//调节色彩的浓淡，即融合比例
    for (int i = 0; i < bmask.rows; i++)
        for (int j = 0; j < bmask.cols; j++)
        {

            tmp = bmask.at<Vec4b>(i, j)[3];
            tmp = 0.6*(tmp / 255.0);
            if (tmp > 1)
                tmp = 1;
            if ((i + origin.y) <= 0 || (j + origin.x) <= 0)
                continue;
            src.at<Vec3b>(i + origin.y, j + origin.x)[0] = int(src.at<Vec3b>(i + origin.y, j + origin.x)[0] * (1 - tmp) +/*dst_image.at<Vec3b>(i, j)[0]*/color[2] * tmp);
            src.at<Vec3b>(i + origin.y, j + origin.x)[1] = int(src.at<Vec3b>(i + origin.y, j + origin.x)[1] * (1 - tmp) + /*dst_image.at<Vec3b>(i, j)[1] */color[1] * tmp);
            src.at<Vec3b>(i + origin.y, j + origin.x)[2] = int(src.at<Vec3b>(i + origin.y, j + origin.x)[2] * (1 - tmp) + /*dst_image.at<Vec3b>(i, j)[2] */color[0] * tmp);
        }
}
//关键点选择使用，上沿42,41,40；43,48,47;
void blush(cv::Mat src, std::vector<cv::Point> points, cv::Mat bmask_left, cv::Mat bmask_right) {
    //计算人脸中轴线及倾斜角度
    std::vector<cv::Point2f> middle_line;
    //nose点位27-30，3D->2D后坐标的问题,人脸倾斜时可能会出现偏差，使用脸廓作为倾斜角
    for (int i = 1, j = 15; i < 7; i++, j--)
        middle_line.push_back((points[i] + points[j]) / 2);
    cv::Vec4f line;
    cv::fitLine(middle_line, line, CV_DIST_L1, 0, 0.01, 0.01);//返回的是弧度值

    if (line[1] > 0)
        line[1] = -line[1];//接近直角下，拟合曲线角度会出现在下象限
    float angle = std::atan2(line[1], line[0]);//弧度角
    //腮红域方框拟合，选定矩形边界
    //左边界,top,points[2],points[3],points[39]
    cv::Point2f top = (points[40] + points[41]) / 2;
    cv::Point2f t1 = lineIntersection(points[1], top, angle);
    cv::Point2f t2 = lineIntersection(points[39], points[3], angle);
    cv::Point2f t3 = lineIntersection(t1, t2, angle);
    //右边界
    cv::Point2f top_r = (points[46] + points[47]) / 2;
    cv::Point2f t1_r = lineIntersection(points[15], top_r, angle);
    cv::Point2f t2_r = lineIntersection(points[42], points[13], angle);
    cv::Point2f t3_r = lineIntersection(t1_r, t2_r, angle);
    matrix_calculate(t1, t2, t3, bmask_left, src, angle, 1);
    matrix_calculate(t1_r, t2_r, t3_r, bmask_right, src, angle, 0);
}