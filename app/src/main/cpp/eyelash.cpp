//
// Created by 陶然 on 2018/12/4.
//
#include "eyelash.h"
#include "ImageWarp.h"
#include <android/log.h>


template<typename T>
inline T rad2deg(const T &radians) {
    static_assert(std::is_floating_point<T>::value, "limited to floating-point type only");
    return radians * (180 / static_cast<T>(M_PI));
}

//混合模板和颜色值
cv::Mat pack(const cv::Mat &mask, uint32_t color) {
    assert(mask.type() == CV_8UC1);
    cv::Mat image(mask.rows, mask.cols, CV_8UC4);

    const int length = mask.rows * mask.cols;
    const uint8_t *mask_data = mask.data;
    uint32_t *image_data = reinterpret_cast<uint32_t *>(image.data);

#pragma omp parallel for
    for (int i = 0; i < length; ++i) {
        uint8_t alpha = ((color >> 24) * mask_data[i] + 127) / 255;
#if USE_BGRA_LAYOUT
        // Swap R and B channel, then assembly it to BGRA format.
            image_data[i] = ((color >> 16) & 0xFF) | (color &0x00FF00) | ((color & 0xFF) << 16) | (alpha << 24);
#else
        image_data[i] = (color & 0x00FFFFFF) | (alpha << 24);
#endif
    }

    return image;
}

cv::Point2f transform(const cv::Mat &affine, const cv::Point2f &point) {
    assert(affine.rows == 2 && affine.cols == 3);
    const float *m = affine.ptr<float>();
    return cv::Point2f(
            m[0] * point.x + m[1] * point.y + m[2],
            m[3] * point.x + m[4] * point.y + m[5]);
}

cv::Mat transform(cv::Size &size, cv::Point2f &pivot, const float &angle,
                  const cv::Vec2f &scale/* = cv::Vec2f(1.0f, 1.0f) */) {
    //模板大小  模板中心点   原图的中心点角度   （原图左点到中心点距离与模板距离的比例）
    /*
	@see http://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
		cv::Mat matrix = cv::getRotationMatrix2D(pivot, angle, scale);
		cv::Rect rect = cv::RotatedRect(pivot, size, angle).boundingRect();

	getRotationMatrix2D is fine, but I need to tweak this function for anisotropical scaling.
	Y is top down, so a clockwise rotating `angle` means a counter-clockwise rotating `-angle` in Y bottom up coordinate.

	/ x' \  = / cos(angle)  -sin(angle) \ . / scale[0]     0     \ . / 1 0  -pivot.x \ . / x \
	\ y' /    \ sin(angle)   cos(angle) /   \    0      scale[1] /   \ 0 1  -pivot.y /   \ y /

	let
		ca_s0 = cos(angle) * scale[0];  sa_s1 = sin(angle) * scale[1];
		sa_s0 = sin(angle) * scale[0];  ca_s1 = cos(angle) * scale[1];

	and the affine matrix is:

		/ ca_s0  -sa_s1  -ca_s0*pivot.x+sa_s1*pivot.y \
		\ sa_s0   ca_s1  -sa_s0*pivot.x-ca_s1*pivot.y /
*/

    float c = std::cos(angle), s = std::sin(angle);

    cv::Mat affine(2, 3, CV_32F);
    float *m = affine.ptr<float>();

    m[0] = c * scale[0];
    m[1] = -s * scale[1];
    m[2] = m[0] * -pivot.x + m[1] * -pivot.y;
    m[3] = s * scale[0];
    m[4] = c * scale[1];
    m[5] = m[3] * -pivot.x + m[4] * -pivot.y;

    assert(size.width > 1 && size.height > 1);  // in case underflow
    if (size.width <= 1 && size.height <= 1) {
        m[0] = 1.0f;
        m[1] = 0.0f;
        m[2] = 0.0f;
        m[0] = 0.0f;
        m[1] = 1.0f;
        m[2] = 0.0f;
        return affine;
    }

    float x = (size.width > 1) ? size.width - 1 : 0;
    float y = (size.height > 1) ? size.height - 1 : 0;
    std::vector<cv::Point2f> points{pivot, cv::Point2f(0, 0), cv::Point2f(x, 0), cv::Point2f(0, y),
                                    cv::Point2f(x, y)};
    cv::transform(points, points, affine);

    cv::Rect rect = cv::boundingRect(points);
    pivot = points[0];

    // make all the stuff relative to origin
    m[2] -= rect.x;
    pivot.x -= rect.x;
    m[5] -= rect.y;
    pivot.y -= rect.y;
    size.width = rect.width;
    size.height = rect.height;

    return affine;
}

float distance(const cv::Point2f &pt0, const cv::Point2f &pt1) {
//	return std::hypot(pt0.x - pt1.x, pt0.y - pt1.y);
    float dx = pt0.x - pt1.x;
    float dy = pt0.y - pt1.y;
    return std::sqrt(dx * dx + dy * dy);
}

cv::Vec3b mix(const cv::Vec3b &from, const cv::Vec3b &to, float amount) {
    cv::Vec3b result;
    int a = cvRound(255 * amount);
    int l_a = 255 - a;

    for (int i = 0; i < 3; ++i)
        result[i] = (from[i] * l_a + to[i] * a + 127) / 255;
    return result;
}

cv::Vec4b mix(const cv::Vec4b &from, const cv::Vec4b &to, float amount) {
    cv::Vec4b result;
    int a = cvRound(to[3] * amount);
    int l_a = 255 - a;

    for (int i = 0; i < 3; ++i)
        result[i] = (from[i] * l_a + to[i] * a + 127) / 255;
    result[3] = from[3];
    return result;
}

void blend(cv::Mat &result, const cv::Mat &dst, const cv::Mat &src,
           const cv::Point2i &origin, float amount) {
    assert(!src.empty() && (src.type() == CV_8UC4 || src.type() == CV_32FC4));

    // Note that dst.copyTo(result); will invoke result.create(src.size(), src.type());
    // which has this clause if( dims <= 2 && rows == _rows && cols == _cols && type() == _type && data ) return;
    // which means that result's memory will only be allocated the first time in if result is empty.
    if (dst.data != result.data)
        dst.copyTo(result);

    cv::Rect rect_src(origin.x, origin.y, src.cols, src.rows);
    cv::Rect rect_dst(0, 0, dst.cols, dst.rows);
    cv::Rect rect = rect_dst & rect_src;

    switch (dst.type()) {
        case CV_8UC3:
            for (int r = rect.y, r_end = rect.y + rect.height; r < r_end; ++r)
                for (int c = rect.x, c_end = rect.x + rect.width; c < c_end; ++c) {
                    const cv::Vec4b &src_color = src.at<cv::Vec4b>(r - origin.y, c - origin.x);
                    cv::Vec3b &dst_color = result.at<cv::Vec3b>(r, c);
                    dst_color = mix(dst_color, *reinterpret_cast<const cv::Vec3b *>(&src_color),
                                    src_color[3] / 255.0F * amount);
                }
            break;
        case CV_8UC4:
            for (int r = rect.y, r_end = rect.y + rect.height; r < r_end; ++r)
                for (int c = rect.x, c_end = rect.x + rect.width; c < c_end; ++c) {
                    const cv::Vec4b &src_color = src.at<cv::Vec4b>(r - origin.y, c - origin.x);
                    cv::Vec4b &dst_color = result.at<cv::Vec4b>(r, c);
                    dst_color = mix(dst_color, src_color, amount);
                }
            break;
#if 0  // currently unused case
        case CV_32FC3:
                for(int r = rect.y, r_end = rect.y + rect.height; r < r_end; ++r)
                for(int c = rect.x, c_end = rect.x + rect.width;  c < c_end; ++c)
                {
                    const cv::Vec3f& src_color = src.at<cv::Vec3f>(r - origin.y, c - origin.x);
                    cv::Vec3f& dst_color = result.at<cv::Vec3f>(r, c);
                    dst_color = mix(dst_color, src_color, amount);
                }
                break;
            case CV_32FC4:
                for(int r = rect.y, r_end = rect.y + rect.height; r < r_end; ++r)
                for(int c = rect.x, c_end = rect.x + rect.width;  c < c_end; ++c)
                {
                    const cv::Vec4f& src_color = src.at<cv::Vec4f>(r - origin.y, c - origin.x);
                    cv::Vec4f& dst_color = result.at<cv::Vec4f>(r, c);
                    dst_color = mix(dst_color, src_color, amount);
                }
                break;
#endif
        default:
            assert(false);
            break;
    }
}


void applyEye(cv::Mat &dst, const cv::Mat &src, const std::vector<cv::Point2f> &points,
              const cv::Mat &cosmetic, float amount) {
    assert(src.type() == CV_8UC4 && cosmetic.type() == CV_8UC4);
    if (src.data != dst.data)
        src.copyTo(dst);

/*
	Below are eye feature point indices:


				36                    46
			 37    35              45    47
	right  38   42   34 -------- 44   43   48   left
			 39    41              51    49
				40                    50

*/

    // I rearrange eye lashes into file doc/eye_lash.xcf
//    const std::vector<cv::Point2f> src_points  // corresponding index 34~41
//            {
//                    cv::Point2f(633, 287), cv::Point2f(534, 228), cv::Point2f(458, 213),
//                    cv::Point2f(386, 228),
//                    cv::Point2f(290, 287), cv::Point2f(386, 350), cv::Point2f(458, 362),
//                    cv::Point2f(534, 353),
//            };

    const std::vector<cv::Point2f> src_points  // corresponding index 34~41
            {
                    cv::Point2f(284, 292), cv::Point2f(404, 225), cv::Point2f(528, 229),
                    cv::Point2f(634, 312),
                    cv::Point2f(496, 374), cv::Point2f(378, 362),
            };


    //眼睛点位数量
    constexpr int N = 41 - 36 + 1;
    std::vector<cv::Point2f> dst_points(N);

    auto calcuateEyeParams = [](const cv::Point2f &right, const cv::Point2f &left) -> cv::Vec4f {

        //左右点中心点
        cv::Point2f pivot = (right + left) / 2;

        //左点和中心点的距离
        float radius = distance(pivot, left);

        //从起止点开始的中心点
        cv::Point2f delta = right - left;
        if (delta.x < 0)
            delta = -delta;  // map angle into interval [-pi/2, pi/2]

        //点（x，y）的角度值
        float angle = std::atan2(delta.y, delta.x);

        return cv::Vec4f(pivot.x, pivot.y, radius, angle);
    };

    //0,4 眼睛点位左右点  素材眼睛参数
    const cv::Vec4f PARAMS = calcuateEyeParams(src_points[0], src_points[3]);

    for (int j = 0; j < 2; ++j) {
        const bool right = (j == 0);
        const int START = right ? 36 : 42;

        // for right: 34 35 36 37 38 39 40 41, formular 34 + i;
        // for left : 48 47 46 45 44 51 50 49, formular 44 + (12 - i)%8;
        if (right)
            for (int i = 0; i < N; ++i)
                dst_points[i] = points[36 + i];
        else {
            const float sum = points[42].x + points[45].x;  // only flip horizontally
            for (int i = 0; i < N; ++i)
                dst_points[i] = cv::Point2f(sum - points[42 + i].x, points[42 + i].y);
        }
        //原图眼睛参数
        cv::Vec4f params = calcuateEyeParams(dst_points[0], dst_points[3]);
        printf("pivot: (%f, %f), radius: %f, angle: %f\n", params[0], params[1], params[2],
               rad2deg(params[3]));

        cv::Size size(cosmetic.cols, cosmetic.rows);
        cv::Point2f pivot(PARAMS[0], PARAMS[1]);
        float angle = params[3];
        float scale = params[2] / PARAMS[2];  //根据原图和素材图中心点和左点的距离，算出两图的大小比例

        //获得旋转矩阵  模板大小  模板中心点   原图的中心点角度   （原图左点到中心点距离与模板距离的比例）
        cv::Mat affine = transform(size, pivot, angle, cv::Point2f(scale, scale));
        //对素材图进行仿射变换
        cv::Mat _cosmetic;
        cv::warpAffine(cosmetic, _cosmetic, affine, size, cv::INTER_LANCZOS4,
                       cv::BORDER_CONSTANT);
        //获取变换后的眼睛关键点，中位点
        std::vector<cv::Point2f> affined_src_points;
        cv::transform(src_points, affined_src_points, affine);
        pivot = transform(affine, cv::Point2f(PARAMS[0], PARAMS[1]));

        // and then move points to make src_points and dst_points' pivots coincide.
        const cv::Point2f dst_pivot = (points[START] + points[START + 3]) / 2;
        const cv::Point2f offset = (affined_src_points[0] + affined_src_points[3]) / 2 - dst_pivot;
        for (size_t i = 0; i < N; ++i)
            dst_points[i] = dst_points[i] + offset;
        //dst_points 目标图眼睛点位  affined_src_points 素材图眼睛点位   素材图mat
        venus::ImageWarp_Rigid warp;
        warp.setMappingPoints(dst_points, affined_src_points);
        warp.setSourceSize(_cosmetic.cols, _cosmetic.rows);
        warp.setTargetSize(_cosmetic.cols, _cosmetic.rows);
        warp.calculateDelta(1.0F);
        _cosmetic = warp.genNewImage(_cosmetic, 1.0F);
//		cv::imshow("test" + std::to_string(j), _cosmetic);

        if (!right) {
            // NOTICE the -1 here, since left(0) + right(cols - 1) == cols - 1.
            pivot.x = static_cast<float>(_cosmetic.cols - 1) - pivot.x;
            cv::flip(_cosmetic, _cosmetic, 1/* horizontally */);
        }
        cv::Point2i origin = dst_pivot - pivot;

        blend(dst, dst, _cosmetic, origin, amount);
    }

}

//void applyEyeLash(cv::Mat &dst, const cv::Mat &src, const std::vector<cv::Point2f> &points,
//                  const cv::Mat &mask, uint32_t color, float amount) {
//    assert(mask.type() == CV_8UC1);
//    cv::Mat eye_lash = pack(mask, color);
//    applyEye(dst, src, points, eye_lash, amount);
//}

void applyEyeLash(cv::Mat &dst, const cv::Mat &src, const std::vector<cv::Point2f> &points,
                  const cv::Mat &mask, float amount) {
    assert(mask.type() == CV_8UC1);
    cv::Mat eye_lash = pack(mask, 4279374609);
    bool isfourdst = dst.type() == CV_8UC4;
    bool isfourdsrc = src.type() == CV_8UC4;
    bool isfourmask = eye_lash.type() == CV_8UC4;
    __android_log_print(ANDROID_LOG_INFO, "taoran", "dst.type() == CV_8UC4  %d ", isfourdst);
    __android_log_print(ANDROID_LOG_INFO, "taoran", "src.type() == CV_8UC4  %d ", isfourdsrc);
    __android_log_print(ANDROID_LOG_INFO, "taoran", "eye_lash.type() == CV_8UC4  %d ", isfourmask);
    applyEye(dst, src, points, eye_lash, amount);

}

void eyelash(cv::Mat &dst, cv::Mat lash, std::vector<cv::Point> shape_points) {

    cv::Mat lashSrc;
    dst.copyTo(lashSrc);

    __android_log_print(ANDROID_LOG_INFO, "taoran", "lash.cols %d lash.cols  %d ", lash.cols,
                        lash.rows);
    std::vector<cv::Point2f> points;

    for (int i = 0; i < shape_points.size(); i++) {
        points.push_back(cv::Point(shape_points[i].x, shape_points[i].y));
    }

    points[45] = cv::Point(shape_points[42].x, shape_points[42].y);
    points[44] = cv::Point(shape_points[43].x, shape_points[43].y);
    points[43] = cv::Point(shape_points[44].x, shape_points[44].y);
    points[42] = cv::Point(shape_points[45].x, shape_points[45].y);
    points[47] = cv::Point(shape_points[46].x, shape_points[46].y);
    points[46] = cv::Point(shape_points[47].x, shape_points[47].y);

    applyEyeLash(dst, lashSrc, points, lash, 1);

}



