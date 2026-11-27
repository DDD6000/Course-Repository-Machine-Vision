// 图像倾斜校正(旋转正) —— 木桌上的彩色书本
//
// 思路与 homework3 相同: 把"倾斜的书"当成一块区域来找姿态
//   颜色分割出书本 -> 形态学 + 最大连通域 -> findContours + minAreaRect
//   -> 由最小外接矩形的旋转角求出需要转的角度 -> warpAffine 旋转 -> 裁剪出摆正的图
//   -> 用同样的方法在旋转后的图上复测残余角度，必要时再修正一次(迭代精化)
//
// 与 homework3 的唯一区别在"怎么把目标分割出来"：
//   作业3 是白底扫描件，用固定阈值取"非白"；
//   这张图背景是米黄色木纹、书本是蓝色，所以在 Lab 色彩空间的 b*(黄-蓝轴)上用 Otsu 分割
//   —— b* 值大=偏黄(木桌)，b* 值小=偏蓝(书本)，比在灰度上用阈值稳得多。
//
// 编译: 调用 vcvars64.bat 后用 cl 编译，链接 opencv_world4120.lib
// 运行: deskew.exe [图片路径]   默认 image.jpg

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// 把中间步骤图存到磁盘
static void saveStep(const std::string& outDir, const std::string& name, const cv::Mat& img) {
    const std::string path = outDir + "/" + name;
    cv::imwrite(path, img);
    std::cout << "[保存] " << path << std::endl;
}

// 把角度折叠到 (-45, 45]:
//   直线没有方向性(0° 与 180° 是同一条)，而且"水平边/竖直边"两族只差 90°，
//   所以只关心它离最近的坐标轴差多少度。
static double foldToAxis(double deg) {
    double d = std::fmod(deg + 45.0, 90.0);
    if (d < 0) d += 90.0;
    return d - 45.0;
}

// 一次测量的结果
struct Measure {
    double angle = 0.0;         // 需要旋转的角度
    cv::RotatedRect rect;       // 目标的最小外接矩形
    cv::Mat componentMask;      // 最大连通域(前景)掩膜
};

// 前景分割: 木桌偏黄(b* 大)、书本偏蓝(b* 小)，在 Lab 的 b* 通道上做 Otsu 自动分割；
// 再乘上"亮度不能太暗"，避免把书本投影的阴影也当成目标
static cv::Mat buildMask(const cv::Mat& bgr, const cv::Mat& gray) {
    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch;
    cv::split(lab, ch);                 // ch[2] 是 b* 通道
    cv::Mat mask;
    cv::threshold(ch[2], mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    cv::Mat bright;
    cv::threshold(gray, bright, 60, 255, cv::THRESH_BINARY);
    cv::bitwise_and(mask, bright, mask);
    return mask;
}

// 形态学去噪 + 只保留最大连通域(即书本本体)
static cv::Mat largestComponent(const cv::Mat& binary) {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::Mat m;
    cv::morphologyEx(binary, m, cv::MORPH_CLOSE, kernel);  // 补上封面上的花纹断口
    cv::morphologyEx(m, m, cv::MORPH_OPEN, kernel);        // 去掉零星噪点

    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(m, labels, stats, centroids, 8, CV_32S);
    int best = -1, bestArea = 0;
    for (int i = 1; i < n; ++i) {
        const int a = stats.at<int>(i, cv::CC_STAT_AREA);
        if (a > bestArea) {
            bestArea = a;
            best = i;
        }
    }
    cv::Mat out = cv::Mat::zeros(m.size(), CV_8U);
    if (best > 0) out.setTo(255, labels == best);
    return out;
}

// 测量一次: 分割 + 最大连通域 + 最小外接矩形 -> 需要旋转的角度
static Measure measure(const cv::Mat& bgr) {
    Measure r;
    cv::Mat gray;
    if (bgr.channels() == 3) cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    else gray = bgr;

    r.componentMask = largestComponent(buildMask(bgr, gray));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(r.componentMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return r;

    size_t best = 0;
    double bestArea = -1.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double a = cv::contourArea(contours[i]);
        if (a > bestArea) {
            bestArea = a;
            best = i;
        }
    }
    r.rect = cv::minAreaRect(contours[best]);
    r.angle = foldToAxis(r.rect.angle);
    return r;
}

// 旋转(画布自动扩大，内容不会被裁掉)，空白处填白
static cv::Mat rotateKeepAll(const cv::Mat& src, double angle) {
    const int w = src.cols, h = src.rows;
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(w / 2.0f, h / 2.0f), angle, 1.0);
    const double cosA = std::fabs(M.at<double>(0, 0));
    const double sinA = std::fabs(M.at<double>(0, 1));
    const int nw = static_cast<int>(h * sinA + w * cosA);
    const int nh = static_cast<int>(h * cosA + w * sinA);
    M.at<double>(0, 2) += nw / 2.0 - w / 2.0;
    M.at<double>(1, 2) += nh / 2.0 - h / 2.0;
    cv::Mat out;
    cv::warpAffine(src, out, M, cv::Size(nw, nh), cv::INTER_CUBIC, cv::BORDER_CONSTANT,
                   cv::Scalar(255, 255, 255));
    return out;
}

int main(int argc, char** argv) {
    const std::string imgPath = (argc > 1) ? argv[1] : "image.jpg";

    std::string outDir = ".";
    const size_t slash = imgPath.find_last_of("/\\");
    if (slash != std::string::npos) outDir = imgPath.substr(0, slash);

    // ---------- 步骤 1: 读入原图 ----------
    cv::Mat src = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图片: " << imgPath << std::endl;
        return -1;
    }
    std::cout << "原图尺寸 = " << src.cols << " x " << src.rows << std::endl;
    saveStep(outDir, "step1_src.png", src);

    // ---------- 步骤 2: 灰度 ----------
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    saveStep(outDir, "step2_gray.png", gray);

    // ---------- 步骤 3: 高斯平滑 ----------
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    saveStep(outDir, "step3_blur.png", blur);

    // ---------- 步骤 4: 颜色分割(Lab b* 通道 Otsu) ----------
    cv::Mat mask = buildMask(src, blur);
    saveStep(outDir, "step4_mask.png", mask);

    // ---------- 步骤 5: 形态学 + 最大连通域 ----------
    cv::Mat comp = largestComponent(mask);
    saveStep(outDir, "step5_morph.png", comp);

    // ---------- 步骤 6: 轮廓 + 最小外接矩形, 量出倾角 ----------
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(comp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) {
        std::cerr << "未检测到目标" << std::endl;
        return -1;
    }
    size_t best = 0;
    double bestArea = -1.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double a = cv::contourArea(contours[i]);
        if (a > bestArea) {
            bestArea = a;
            best = i;
        }
    }
    const cv::RotatedRect rect0 = cv::minAreaRect(contours[best]);
    const double angle0 = foldToAxis(rect0.angle);

    cv::Mat rectVis = src.clone();
    cv::drawContours(rectVis, contours, static_cast<int>(best), cv::Scalar(0, 255, 0), 2);
    cv::Point2f corner[4];
    rect0.points(corner);
    for (int i = 0; i < 4; ++i) {
        cv::line(rectVis, corner[i], corner[(i + 1) % 4], cv::Scalar(0, 255, 255), 3, cv::LINE_AA);
        cv::circle(rectVis, corner[i], 5, cv::Scalar(0, 0, 255), -1);
    }
    cv::putText(rectVis, "angle=" + cv::format("%.2f", angle0),
                cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 1.4, cv::Scalar(0, 255, 255), 3);
    saveStep(outDir, "step6_minrect.png", rectVis);

    std::cout << "最小外接矩形: 中心=(" << cvRound(rect0.center.x) << "," << cvRound(rect0.center.y)
              << ")  尺寸=" << cvRound(rect0.size.width) << "x" << cvRound(rect0.size.height)
              << "  minAreaRect 角=" << cv::format("%.2f", rect0.angle)
              << "  折叠后需旋转=" << cv::format("%.2f", angle0) << "°" << std::endl;

    // ---------- 步骤 7: 迭代旋转校正 ----------
    // 一次旋转后重新测量、必要时再修一次：
    // 单次测量受量化误差和插值影响，迭代两三次能把残余角压到 0.25° 以内
    cv::Mat cur = src;
    double totalAngle = 0.0;
    for (int it = 0; it < 3; ++it) {
        const Measure m = measure(cur);
        if (std::fabs(m.angle) < 0.25) {
            std::cout << "  迭代 " << it << ": 残余 " << cv::format("%.2f", m.angle) << "°，收敛" << std::endl;
            break;
        }
        std::cout << "  迭代 " << it << ": 修正 " << cv::format("%.2f", m.angle) << "°" << std::endl;
        cur = rotateKeepAll(cur, m.angle);
        totalAngle += m.angle;
    }
    saveStep(outDir, "step7_rotated.png", cur);

    // ---------- 步骤 8: 裁剪到目标外接矩形 ----------
    // 旋转后矩形已经与坐标轴对齐，直接取它的外接框即可裁掉四周空白
    const Measure fin = measure(cur);
    cv::Rect box = fin.rect.boundingRect() & cv::Rect(0, 0, cur.cols, cur.rows);
    cv::Mat cropped = cur(box).clone();
    saveStep(outDir, "step8_cropped.png", cropped);

    std::cout << "----" << std::endl;
    std::cout << "总旋转角 = " << cv::format("%.2f", totalAngle) << "°，残余倾角 = "
              << cv::format("%.2f", fin.angle) << "°" << std::endl;
    std::cout << "校正后尺寸 = " << cropped.cols << " x " << cropped.rows << std::endl;
    return 0;
}
