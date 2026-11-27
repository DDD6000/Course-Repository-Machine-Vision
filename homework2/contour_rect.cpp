// 练习4: 轮廓提取 + 最小外接矩形(姿态估计)
// 对应练习截图: src 窗口里零件被黄色轮廓描边，外面套一个斜置的黄色矩形
//
// 处理流程:
//   读原图 -> (过小时等比放大) -> 灰度 -> 高斯平滑 -> Otsu 二值化(极性自适应)
//   -> 形态学去噪 -> 轮廓提取(RETR_CCOMP, 含内孔) -> 最小外接矩形 minAreaRect
//   -> 输出中心点 / 长宽尺寸 / 倾斜角度
//
// 编译: 调用 vcvars64.bat 后用 cl 编译，链接 opencv_world4120.lib
// 运行: contour_rect.exe [图片路径]   默认 image.jpg

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

// 把中间步骤图存到磁盘
static void saveStep(const std::string& outDir, const std::string& name, const cv::Mat& img) {
    const std::string path = outDir + "/" + name;
    cv::imwrite(path, img);
    std::cout << "[保存] " << path << std::endl;
}

// 极性自适应: 背景一般占据图像边框，统计四条边上白色像素的比例，
// 若边框大部分是白色，说明白色其实是背景，需要把二值图反相，保证"前景=目标"
static void ensureForegroundIsWhite(cv::Mat& binary) {
    const int w = binary.cols, h = binary.rows;
    int borderPix = 0, borderWhite = 0;
    for (int x = 0; x < w; ++x) {
        borderWhite += (binary.at<uchar>(0, x) > 0) + (binary.at<uchar>(h - 1, x) > 0);
        borderPix += 2;
    }
    for (int y = 0; y < h; ++y) {
        borderWhite += (binary.at<uchar>(y, 0) > 0) + (binary.at<uchar>(y, w - 1) > 0);
        borderPix += 2;
    }
    if (borderWhite * 2 > borderPix) {
        cv::bitwise_not(binary, binary);
        std::cout << "  [极性自适应] 边框以白色为主，二值图已反相" << std::endl;
    }
}

int main(int argc, char** argv) {
    const std::string imgPath = (argc > 1) ? argv[1] : "image.jpg";

    std::string outDir = ".";
    const size_t slash = imgPath.find_last_of("/\\");
    if (slash != std::string::npos) outDir = imgPath.substr(0, slash);

    // ---------- 步骤 1: 读入原始彩色图 ----------
    cv::Mat color = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (color.empty()) {
        std::cerr << "无法读取图片: " << imgPath << std::endl;
        return -1;
    }
    std::cout << "原图尺寸 = " << color.cols << " x " << color.rows << std::endl;
    saveStep(outDir, "step1_src.png", color);

    // ---------- 步骤 2: 图像过小时等比放大 ----------
    // 108x48 这种小图上，轮廓点太少、矩形精度差，先放大再处理
    const int kMinLongSide = 600;
    cv::Mat work = color.clone();
    if (std::max(work.cols, work.rows) < kMinLongSide) {
        const double scale = static_cast<double>(kMinLongSide) / std::max(work.cols, work.rows);
        cv::resize(color, work, cv::Size(), scale, scale, cv::INTER_CUBIC);
        std::cout << "缩放系数 = " << cv::format("%.2f", scale) << std::endl;
    }
    saveStep(outDir, "step2_resize.png", work);

    // ---------- 步骤 3: 彩色转灰度 ----------
    cv::Mat gray;
    cv::cvtColor(work, gray, cv::COLOR_BGR2GRAY);
    saveStep(outDir, "step3_gray.png", gray);

    // ---------- 步骤 4: 高斯平滑 ----------
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    saveStep(outDir, "step4_blur.png", blur);

    // ---------- 步骤 5: Otsu 二值化(极性自适应) ----------
    cv::Mat binary;
    const double otsuTh = cv::threshold(blur, binary, 0, 255,
                                        cv::THRESH_BINARY | cv::THRESH_OTSU);
    ensureForegroundIsWhite(binary);  // 保证"零件是白色前景"
    std::cout << "Otsu 阈值 = " << otsuTh
              << "，前景占比 = " << cv::format("%.3f", cv::countNonZero(binary) / (double)binary.total())
              << std::endl;
    saveStep(outDir, "step5_binary.png", binary);

    // ---------- 步骤 6: 形态学去噪 ----------
    // 闭运算补边缘小缺口(避免轮廓破碎)，开运算去掉零星噪点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    saveStep(outDir, "step6_morph.png", binary);

    // ---------- 步骤 7: 轮廓提取 ----------
    // RETR_CCOMP: 只分两层，外轮廓 + 内孔轮廓，正好能区分"零件外形"和"腰形槽/圆孔"
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
    std::cout << "轮廓总数 = " << contours.size() << std::endl;
    if (contours.empty()) {
        std::cerr << "未检测到轮廓" << std::endl;
        return -1;
    }

    // 面积最大的外轮廓就是零件本体
    int mainIdx = -1;
    double maxArea = -1.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double a = cv::contourArea(contours[i]);
        if (hierarchy[i][3] == -1 && a > maxArea) {  // parent == -1 才是外轮廓
            maxArea = a;
            mainIdx = static_cast<int>(i);
        }
    }
    std::cout << "零件轮廓: 序号 " << mainIdx << "，面积 " << cvRound(maxArea)
              << "，周长 " << cvRound(cv::arcLength(contours[mainIdx], true)) << std::endl;

    // 绘制全部轮廓: 外轮廓黄色、内孔绿色
    cv::Mat contourVis = work.clone();
    for (size_t i = 0; i < contours.size(); ++i) {
        const bool isHole = (hierarchy[i][3] != -1);
        cv::drawContours(contourVis, contours, static_cast<int>(i),
                         isHole ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255),
                         isHole ? 2 : 3);
    }
    saveStep(outDir, "step7_contours.png", contourVis);

    // ---------- 步骤 8: 最小外接矩形 minAreaRect ----------
    // 与 boundingRect 的区别: boundingRect 是水平矩形，零件一斜就会大出一圈；
    // minAreaRect 找面积最小的旋转矩形，顺便给出中心点、长宽、旋转角 —— 即零件的位姿
    const cv::RotatedRect rect = cv::minAreaRect(contours[mainIdx]);
    cv::Point2f corner[4];
    rect.points(corner);  // 四个角点(顺时针)

    // 长边方向角: 用矩形相邻两条边中较长的那条求方位角
    double longEdgeAngle = 0.0;
    double longLen = 0.0, shortLen = 0.0;
    for (int i = 0; i < 2; ++i) {
        const double dx = corner[i + 1].x - corner[i].x;
        const double dy = corner[i + 1].y - corner[i].y;
        const double len = std::hypot(dx, dy);
        if (len > longLen) {
            shortLen = longLen;
            longLen = len;
            longEdgeAngle = std::atan2(dy, dx) * 180.0 / CV_PI;
        } else {
            shortLen = len;
        }
    }
    while (longEdgeAngle <= -90.0) longEdgeAngle += 180.0;
    while (longEdgeAngle > 90.0) longEdgeAngle -= 180.0;

    cv::Mat rectVis = work.clone();
    for (int i = 0; i < 4; ++i) {
        cv::line(rectVis, corner[i], corner[(i + 1) % 4], cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::circle(rectVis, corner[i], 4, cv::Scalar(0, 0, 255), -1);
    }
    cv::circle(rectVis, rect.center, 5, cv::Scalar(0, 0, 255), -1);
    saveStep(outDir, "step8_minrect.png", rectVis);

    // ---------- 步骤 9: 结果图(轮廓 + 矩形 + 中心 + 参数) ----------
    // 内孔个数: 在 hierarchy 里沿着主轮廓的子节点链数一遍
    int holeCount = 0;
    for (int c = hierarchy[mainIdx][2]; c != -1; c = hierarchy[c][0]) ++holeCount;

    cv::Mat result = contourVis.clone();
    for (int i = 0; i < 4; ++i) {
        cv::line(result, corner[i], corner[(i + 1) % 4], cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
    // 中心十字线
    const int cross = 20;
    cv::line(result, cv::Point(cvRound(rect.center.x) - cross, cvRound(rect.center.y)),
             cv::Point(cvRound(rect.center.x) + cross, cvRound(rect.center.y)),
             cv::Scalar(0, 0, 255), 1);
    cv::line(result, cv::Point(cvRound(rect.center.x), cvRound(rect.center.y) - cross),
             cv::Point(cvRound(rect.center.x), cvRound(rect.center.y) + cross),
             cv::Scalar(0, 0, 255), 1);

    // 半透明黑底，保证文字在亮零件和暗背景上都看得清
    cv::Mat panel = result.clone();
    cv::rectangle(panel, cv::Rect(0, 0, 340, 104), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(panel, 0.55, result, 0.45, 0.0, result);

    char info[256];
    std::snprintf(info, sizeof(info), "center=(%d,%d)", cvRound(rect.center.x), cvRound(rect.center.y));
    cv::putText(result, info, cv::Point(12, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    std::snprintf(info, sizeof(info), "size=%.0fx%.0f  angle=%.1fdeg", longLen, shortLen, longEdgeAngle);
    cv::putText(result, info, cv::Point(12, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    std::snprintf(info, sizeof(info), "area=%d  holes=%d", cvRound(maxArea), holeCount);
    cv::putText(result, info, cv::Point(12, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    saveStep(outDir, "step9_result.png", result);

    std::cout << "----" << std::endl;
    std::cout << "最小外接矩形: 中心=(" << cvRound(rect.center.x) << "," << cvRound(rect.center.y) << ")"
              << "  长边=" << cvRound(longLen) << "  短边=" << cvRound(shortLen)
              << "  长边方向角=" << cv::format("%.2f", longEdgeAngle) << "°"
              << "  内孔数=" << holeCount << std::endl;
    return 0;
}
