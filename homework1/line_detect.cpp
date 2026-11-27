// 练习3: Canny 边缘检测 + 霍夫直线检测
// 对应练习截图: 左窗口 canny(边缘图)，右窗口 src(检测到的直线叠加在原图上)
//
// 处理流程:
//   读原图 -> (过小时等比放大) -> 灰度 -> 高斯平滑 -> Canny 边缘检测
//   -> 概率霍夫变换(线段) -> 标准霍夫变换(整条直线) -> 选出主干直线并叠加显示
//
// 所有距离类参数都按图像对角线长度归一化，换分辨率不用重新调参
//
// 编译: 调用 vcvars64.bat 后用 cl 编译，链接 opencv_world4120.lib
// 运行: line_detect.exe [图片路径]   默认 image.jpg

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

// 用直方图求中位数，Canny 的自适应双阈值要用它
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

// 由 rho/theta 形式的直线算出图像内的一段端点(用于绘制)
static void lineEndpoints(double rho, double theta, cv::Point& p1, cv::Point& p2) {
    const double a = std::cos(theta);
    const double b = std::sin(theta);
    const double x0 = a * rho;
    const double y0 = b * rho;
    const double len = 2000.0;  // 足够长，cv::line 会自动裁剪到图像内
    p1 = cv::Point(cvRound(x0 + len * (-b)), cvRound(y0 + len * a));
    p2 = cv::Point(cvRound(x0 - len * (-b)), cvRound(y0 - len * a));
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
    // Canny 的梯度核、霍夫的分辨率和长度阈值都有最小尺度要求；
    // 图太小(如 108x48)时边缘只有几像素长，直线检测结果全是碎片，先放大再处理。
    const int kMinLongSide = 600;
    cv::Mat work = color.clone();
    double scale = 1.0;
    if (std::max(work.cols, work.rows) < kMinLongSide) {
        scale = static_cast<double>(kMinLongSide) / std::max(work.cols, work.rows);
        cv::resize(color, work, cv::Size(), scale, scale, cv::INTER_CUBIC);
    }
    std::cout << "缩放系数 = " << cv::format("%.2f", scale)
              << "，处理尺寸 = " << work.cols << " x " << work.rows << std::endl;
    saveStep(outDir, "step2_resize.png", work);

    // ---------- 步骤 3: 彩色转灰度 ----------
    cv::Mat gray;
    cv::cvtColor(work, gray, cv::COLOR_BGR2GRAY);
    saveStep(outDir, "step3_gray.png", gray);

    // ---------- 步骤 4: 高斯平滑 ----------
    // Canny 对噪声非常敏感，不平滑会把纹理全变成边缘
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    saveStep(outDir, "step4_blur.png", blur);

    // ---------- 步骤 5: Canny 边缘检测 ----------
    // 双阈值由中位数自适应给出(0.66*med / 1.33*med)，比写死 50/150 更通用:
    // 高阈值以上的强边缘必留，低阈值以下的弱边缘必丢，
    // 介于两者之间的弱边缘只有和强边缘相连时才保留(这就是 Canny 的"边缘连接")。
    const double med = medianOf(blur);
    const double lowTh = std::max(0.0, 0.66 * med);
    const double highTh = std::min(255.0, 1.33 * med);
    cv::Mat edges;
    cv::Canny(blur, edges, lowTh, highTh);
    const int edgeCount = cv::countNonZero(edges);
    std::cout << "Canny 阈值 = [" << cvRound(lowTh) << ", " << cvRound(highTh)
              << "]，边缘像素 = " << edgeCount << std::endl;
    saveStep(outDir, "step5_canny.png", edges);

    // 距离类参数统一按对角线归一化
    const double diag = std::hypot(static_cast<double>(work.cols), static_cast<double>(work.rows));

    // ---------- 步骤 6: 概率霍夫变换 HoughLinesP(得到线段) ----------
    // 与标准霍夫的区别: 它顺便给出线段端点，能直接看出每条线的实际长度和位置
    std::vector<cv::Vec4i> segments;
    cv::HoughLinesP(edges, segments,
                    1.0,                 // 距离分辨率 1 像素
                    CV_PI / 180.0,       // 角度分辨率 1 度
                    cvRound(0.05 * diag),// 累加器阈值: 至少这么多个点共线才算直线
                    static_cast<double>(cvRound(0.10 * diag)),  // 线段最短长度
                    static_cast<double>(cvRound(0.02 * diag))); // 允许的最大断裂间隙
    std::cout << "HoughLinesP 线段数 = " << segments.size() << std::endl;

    cv::Mat segVis = work.clone();
    for (size_t i = 0; i < segments.size(); ++i) {
        const cv::Vec4i& s = segments[i];
        cv::line(segVis, cv::Point(s[0], s[1]), cv::Point(s[2], s[3]),
                 cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        const double len = std::hypot(s[2] - s[0], s[3] - s[1]);
        const double ang = std::atan2(static_cast<double>(s[3] - s[1]),
                                      static_cast<double>(s[2] - s[0])) * 180.0 / CV_PI;
        std::cout << "  线段 " << i << " 长度=" << cvRound(len)
                  << "  角度=" << cv::format("%.1f", ang) << "°" << std::endl;
    }
    saveStep(outDir, "step6_hough_segments.png", segVis);

    // ---------- 步骤 7: 标准霍夫变换 HoughLines(得到整条直线的 rho/theta) ----------
    // 极坐标表示: rho = x*cos(theta) + y*sin(theta)，结果按累加器票数从高到低排列
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, 1.0, CV_PI / 180.0, cvRound(0.10 * diag));
    std::cout << "HoughLines 直线数 = " << lines.size() << std::endl;

    cv::Mat lineVis = work.clone();
    for (size_t i = 0; i < lines.size(); ++i) {
        cv::Point p1, p2;
        lineEndpoints(lines[i][0], lines[i][1], p1, p2);
        cv::line(lineVis, p1, p2, cv::Scalar(255, 128, 0), 1, cv::LINE_AA);
    }
    saveStep(outDir, "step7_hough_lines.png", lineVis);

    // ---------- 步骤 8: 选出票数最高的两条作为主干直线，黄色叠加 ----------
    // 若前两条几乎重合(theta 与 rho 都很接近)，跳过重复的那条取下一个，
    // 这样得到的是两条"位置不同"的主线，与练习截图里的效果一致。
    cv::Mat result = work.clone();
    std::vector<cv::Vec2f> main2;
    for (size_t i = 0; i < lines.size(); ++i) {
        bool duplicate = false;
        for (size_t k = 0; k < main2.size(); ++k) {
            const float dTheta = std::fabs(lines[i][1] - main2[k][1]) * 180.0f / static_cast<float>(CV_PI);
            const double dRho = std::fabs(static_cast<double>(lines[i][0]) - main2[k][0]);
            if (dTheta < 5.0f && dRho < 0.02 * diag) duplicate = true;
        }
        if (duplicate) continue;
        main2.push_back(lines[i]);
        if (main2.size() == 2) break;
    }

    for (size_t i = 0; i < main2.size(); ++i) {
        cv::Point p1, p2;
        lineEndpoints(main2[i][0], main2[i][1], p1, p2);
        cv::clipLine(work.size(), p1, p2);
        cv::line(result, p1, p2, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        // 直线方向角 = theta - 90°(theta 是法线方向)
        double lineAngle = main2[i][1] * 180.0 / CV_PI - 90.0;
        if (lineAngle < -90.0) lineAngle += 180.0;
        std::cout << "  主线 " << i + 1 << ": rho=" << cvRound(main2[i][0])
                  << " theta=" << cv::format("%.1f", main2[i][1] * 180.0 / CV_PI)
                  << "° 方向角=" << cv::format("%.1f", lineAngle) << "°" << std::endl;

        // 标签贴着自己那条线，且始终留在图像内(上方的线往上写，下方的线往下写)
        const int midY = (p1.y + p2.y) / 2;
        const int labelY = cv::min(cv::max(midY + (midY < work.rows / 2 ? 24 : -10), 20), work.rows - 10);
        cv::putText(result, "line" + std::to_string(i + 1) + " " + cv::format("%.1f", lineAngle) + "deg",
                    cv::Point(12, labelY), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2);
    }

    // 两条主线的夹角(取锐角)
    if (main2.size() == 2) {
        double a1 = main2[0][1] * 180.0 / CV_PI;
        double a2 = main2[1][1] * 180.0 / CV_PI;
        double diff = std::fabs(a1 - a2);
        if (diff > 90.0) diff = 180.0 - diff;
        std::cout << "  两条主线夹角 = " << cv::format("%.1f", diff) << "°" << std::endl;
    }

    cv::putText(result, "main lines = " + std::to_string(main2.size()),
                cv::Point(12, work.rows - 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    saveStep(outDir, "step8_result.png", result);

    std::cout << "----" << std::endl;
    std::cout << "边缘像素 " << edgeCount << "，线段 " << segments.size()
              << " 条，直线 " << lines.size() << " 条，主干直线 " << main2.size() << " 条" << std::endl;
    return 0;
}
