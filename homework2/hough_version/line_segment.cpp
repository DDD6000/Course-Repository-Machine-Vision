// 练习4: 零件直线边缘检测 (Canny + 概率霍夫变换 HoughLinesP)
// 对应练习截图: src 窗口里零件的直线边被画上黄色线段，圆弧端和两个圆孔没有任何标注
//   —— 这正是"只检测直线段"的特征(轮廓检测会把圆弧和圆孔也描出来)
//
// 处理流程:
//   读原图 -> (过小时等比放大) -> 灰度 -> 高斯平滑 -> Canny 边缘检测
//   -> HoughLinesP 概率霍夫(得到直线段) -> 共线线段合并 -> 主方向统计 -> 叠加显示
//
// 编译: 调用 vcvars64.bat 后用 cl 编译，链接 opencv_world4120.lib
// 运行: line_segment.exe [图片路径]   默认 image.jpg

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

// 一条直线段(带缓存的角度和长度)
struct Seg {
    cv::Point2f a;
    cv::Point2f b;
    double len = 0.0;
    double ang = 0.0;  // 方向角，范围 [0,180)
};

// 把中间步骤图存到磁盘
static void saveStep(const std::string& outDir, const std::string& name, const cv::Mat& img) {
    const std::string path = outDir + "/" + name;
    cv::imwrite(path, img);
    std::cout << "[保存] " << path << std::endl;
}

// 用 256 桶直方图求中位数，给 Canny 的双阈值做自适应
static double medianOf(const cv::Mat& gray) {
    int hist[256] = {0};
    for (int y = 0; y < gray.rows; ++y) {
        const uchar* p = gray.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x) ++hist[p[x]];
    }
    const int total = gray.rows * gray.cols;
    int acc = 0;
    for (int i = 0; i < 256; ++i) {
        acc += hist[i];
        if (acc >= total / 2) return static_cast<double>(i);
    }
    return 128.0;
}

// 方向角统一到 [0,180): 一条直线正反两个方向是同一件事，不需要区分
static double normAngle(double deg) {
    deg = std::fmod(deg, 180.0);
    if (deg < 0) deg += 180.0;
    return deg;
}

static Seg makeSeg(const cv::Point2f& a, const cv::Point2f& b) {
    Seg s;
    s.a = a;
    s.b = b;
    s.len = std::hypot(static_cast<double>(b.x - a.x), static_cast<double>(b.y - a.y));
    s.ang = normAngle(std::atan2(static_cast<double>(b.y - a.y),
                                 static_cast<double>(b.x - a.x)) * 180.0 / CV_PI);
    return s;
}

// 两个角度之间的最小差值(考虑 180° 周期性)
static double angleDiff(double a, double b) {
    double d = std::fabs(a - b);
    if (d > 90.0) d = 180.0 - d;
    return d;
}

// 共线线段合并:
// 同一条边常被检成好几段(受 maxLineGap、边缘断裂影响)，合并后一遍扫描就能看出零件的每条直边
// 判据: 方向接近 + 到基准直线的垂距很小 + 沿方向上的间隔足够小
static std::vector<Seg> mergeSegments(const std::vector<Seg>& src, double diag) {
    const double kAngleTol = 6.0;        // 角度容差(度)
    const double kPerpTol = 0.015 * diag; // 垂距容差
    const double kGapTol = 0.05 * diag;   // 沿方向的间隔容差

    std::vector<int> order(src.size());
    for (size_t i = 0; i < src.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&src](int i, int j) { return src[i].len > src[j].len; });

    std::vector<bool> used(src.size(), false);
    std::vector<Seg> out;

    for (size_t oi = 0; oi < order.size(); ++oi) {
        const int i = order[oi];
        if (used[i]) continue;
        used[i] = true;

        const Seg& base = src[i];
        const double rad = base.ang * CV_PI / 180.0;
        const double ux = std::cos(rad), uy = std::sin(rad);
        std::vector<cv::Point2f> pts{base.a, base.b};

        for (size_t oj = 0; oj < order.size(); ++oj) {
            const int j = order[oj];
            if (used[j]) continue;
            const Seg& s = src[j];
            if (angleDiff(s.ang, base.ang) > kAngleTol) continue;

            // 垂距: 用另一个线段中点到基准直线的距离判断
            const double mx = (s.a.x + s.b.x) * 0.5;
            const double my = (s.a.y + s.b.y) * 0.5;
            const double perp = std::fabs(-uy * (mx - base.a.x) + ux * (my - base.a.y));
            if (perp > kPerpTol) continue;

            // 沿基准方向的投影间隔(相对基准段自身的 [0, len] 区间)
            const double p0 = ux * (s.a.x - base.a.x) + uy * (s.a.y - base.a.y);
            const double p1 = ux * (s.b.x - base.a.x) + uy * (s.b.y - base.a.y);
            const double gap = std::max(std::min(p0, p1) - base.len, -std::max(p0, p1));
            if (gap > kGapTol) continue;

            used[j] = true;
            pts.push_back(s.a);
            pts.push_back(s.b);
        }

        // 把组内所有端点投影到基准方向，取最远两端作为合并结果
        double tmin = 1e18, tmax = -1e18;
        for (const cv::Point2f& p : pts) {
            const double t = ux * (p.x - base.a.x) + uy * (p.y - base.a.y);
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
        }
        out.push_back(makeSeg(cv::Point2f(static_cast<float>(base.a.x + ux * tmin),
                                          static_cast<float>(base.a.y + uy * tmin)),
                              cv::Point2f(static_cast<float>(base.a.x + ux * tmax),
                                          static_cast<float>(base.a.y + uy * tmax))));
    }
    return out;
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
    // Canny 的 3x3 梯度核和霍夫的长度阈值都有最小尺度要求，108x48 这种小图上边缘只有几像素
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

    // ---------- 步骤 5: Canny 边缘检测 ----------
    // 双阈值按中位数自适应: [0.66*med, 1.33*med]
    const double med = medianOf(blur);

    cv::Mat edges;
    const double lowTh = std::max(0.0, 0.66 * med);
    const double highTh = std::min(255.0, 1.33 * med);
    cv::Canny(blur, edges, lowTh, highTh);
    std::cout << "Canny 阈值 = [" << cvRound(lowTh) << ", " << cvRound(highTh)
              << "]，边缘像素 = " << cv::countNonZero(edges) << std::endl;
    saveStep(outDir, "step5_canny.png", edges);

    // ---------- 步骤 6: 概率霍夫变换 HoughLinesP ----------
    // 与 HoughLines(整条无限直线)不同，HoughLinesP 输出的是"有端点、有长度"的线段，
    // 所以短边检不到、圆弧检不到 —— 这就是截图里只有部分直线被标黄的原因
    const double diag = std::hypot(static_cast<double>(work.cols), static_cast<double>(work.rows));
    std::vector<cv::Vec4i> raw;
    cv::HoughLinesP(edges, raw,
                    1.0,
                    CV_PI / 180.0,
                    cvRound(0.05 * diag),   // 累加器阈值(至少多少点共线)
                    cvRound(0.08 * diag),   // 线段最短长度
                    cvRound(0.04 * diag));  // 允许断裂间隙
    std::cout << "HoughLinesP 原始线段 = " << raw.size() << " 条" << std::endl;

    std::vector<Seg> rawSegs;
    for (size_t i = 0; i < raw.size(); ++i) {
        rawSegs.push_back(makeSeg(cv::Point2f((float)raw[i][0], (float)raw[i][1]),
                                  cv::Point2f((float)raw[i][2], (float)raw[i][3])));
    }

    cv::Mat rawVis = work.clone();
    for (const Seg& s : rawSegs) {
        cv::line(rawVis, s.a, s.b, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        std::cout << "  原始线段 长度=" << cvRound(s.len)
                  << "  角度=" << cv::format("%.1f", s.ang) << "°" << std::endl;
    }
    saveStep(outDir, "step6_houghp_raw.png", rawVis);

    // ---------- 步骤 7: 共线线段合并 ----------
    std::vector<Seg> merged = mergeSegments(rawSegs, diag);
    std::sort(merged.begin(), merged.end(), [](const Seg& x, const Seg& y) { return x.len > y.len; });
    std::cout << "合并后线段 = " << merged.size() << " 条" << std::endl;

    cv::Mat mergeVis = work.clone();
    for (const Seg& s : merged) {
        cv::line(mergeVis, s.a, s.b, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
    saveStep(outDir, "step7_houghp_merged.png", mergeVis);

    // ---------- 步骤 8: 主方向统计 + 结果图 ----------
    // 以线长为权重统计方向直方图(1° 一档)，峰值方向即零件主方向；
    // 与主方向接近 90° 的是零件的短边方向
    std::vector<double> hist(180, 0.0);
    for (const Seg& s : merged) {
        const int bin = static_cast<int>(s.ang) % 180;
        hist[bin] += s.len;
    }
    int peakBin = 0;
    for (int i = 0; i < 180; ++i) {
        if (hist[i] > hist[peakBin]) peakBin = i;
    }
    const double mainDir = static_cast<double>(peakBin);

    cv::Mat result = work.clone();
    for (const Seg& s : merged) {
        cv::line(result, s.a, s.b, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
    // 半透明黑底信息面板
    cv::Mat panel = result.clone();
    cv::rectangle(panel, cv::Rect(0, 0, 430, 108), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(panel, 0.55, result, 0.45, 0.0, result);

    char info[256];
    std::snprintf(info, sizeof(info), "raw segs = %d   merged = %d", (int)raw.size(), (int)merged.size());
    cv::putText(result, info, cv::Point(12, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    std::snprintf(info, sizeof(info), "main dir = %.1f deg", mainDir);
    cv::putText(result, info, cv::Point(12, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    std::snprintf(info, sizeof(info), "perp dir = %.1f deg", normAngle(mainDir + 90.0));
    cv::putText(result, info, cv::Point(12, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    saveStep(outDir, "step8_result.png", result);

    std::cout << "----" << std::endl;
    std::cout << "主方向 = " << cv::format("%.1f", mainDir) << "°，垂直方向 = "
              << cv::format("%.1f", normAngle(mainDir + 90.0)) << "°" << std::endl;
    std::cout << "合并后最长线段 = " << cvRound(merged.empty() ? 0.0 : merged[0].len) << " px" << std::endl;
    return 0;
}
