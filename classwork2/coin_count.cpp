// 硬币计数 (coin counting)
// 处理流程: 彩色->灰度 -> 高斯平滑 -> 三角形法二值化 -> 形态学 -> 连通域 -> 面积/边框过滤 -> 计数
// 每一步的中间结果都会保存成图片，方便观察算法过程
//
// 与回形针那题的区别: 硬币是"亮目标 + 暗背景"，二值化不用反相；
// 且直方图是"大块暗背景 + 一小撮亮硬币"的强偏斜分布，Otsu 会把阈值抬到 133，
// 偏暗的那枚硬币被切掉一部分(碎裂成几块)，所以这里改用 Triangle(三角形法)阈值 89。
//
// 编译: 调用 vcvars64.bat 后用 cl 编译，链接 opencv_world4120.lib
// 运行: coin_count.exe [图片路径]   默认 origin.png

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// 把中间步骤图存到磁盘(统一放在输出目录下，带序号前缀)
static void saveStep(const std::string& outDir, const std::string& name, const cv::Mat& img) {
    const std::string path = outDir + "/" + name;
    cv::imwrite(path, img);
    std::cout << "[保存] " << path << std::endl;
}

int main(int argc, char** argv) {
    const std::string imgPath = (argc > 1) ? argv[1] : "origin.png";

    // 输出目录 = 输入图片所在目录；找不到时退回当前目录
    std::string outDir = ".";
    const size_t slash = imgPath.find_last_of("/\\");
    if (slash != std::string::npos) outDir = imgPath.substr(0, slash);

    // ---------- 步骤 0: 读入原始彩色图 ----------
    cv::Mat color = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (color.empty()) {
        std::cerr << "无法读取图片: " << imgPath << std::endl;
        return -1;
    }
    saveStep(outDir, "step1_color.png", color);

    // ---------- 步骤 1: 彩色转灰度 ----------
    // 只关心"亮的硬币 vs 暗的桌面"这个亮度差异，颜色信息用不上
    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    saveStep(outDir, "step2_gray.png", gray);

    // ---------- 步骤 2: 高斯平滑 ----------
    // 抑制桌面纹理和颗粒噪声，避免二值化后出现毛刺
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    saveStep(outDir, "step3_blur.png", blur);

    // ---------- 步骤 3: Triangle 阈值二值化 ----------
    // 硬币比桌面亮，正相二值化(THRESH_BINARY)后硬币为白色前景；
    // 用 Triangle 而非 Otsu：直方图严重偏斜时 Otsu 偏向方差大的亮类，
    // 会把偏暗的那枚硬币切开，计数就会偏多或漏检。
    cv::Mat binary;
    const double th = cv::threshold(blur, binary, 0, 255,
                                    cv::THRESH_BINARY | cv::THRESH_TRIANGLE);
    saveStep(outDir, "step4_binary.png", binary);

    // ---------- 步骤 4: 形态学处理 ----------
    // 闭运算补细小缺口，开运算去掉孤立噪点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    saveStep(outDir, "step5_morph.png", binary);

    // ---------- 步骤 5: 连通域标记 ----------
    // 硬币互不接触，一枚硬币 = 一个连通域；8 邻域标记，同时得到面积、外接矩形、质心
    cv::Mat labels, stats, centroids;
    const int num = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

    // 连通域伪彩色图：给每个标号不同颜色，直观检查有没有粘连或碎裂
    cv::Mat labelsU8, labelsVis;
    labels.convertTo(labelsU8, CV_8U, 12);
    cv::applyColorMap(labelsU8, labelsVis, cv::COLORMAP_JET);
    labelsVis.setTo(cv::Vec3b(0, 0, 0), labels == 0);  // 背景涂黑
    saveStep(outDir, "step6_labels.png", labelsVis);

    // ---------- 步骤 6: 过滤 ----------
    // (a) 剔除贴图像边框的连通域(照片/扫描件的边框可能被当成目标)
    // (b) 面积过滤：以中位面积自适应定下限，去掉碎屑噪点
    std::vector<int> keep;
    std::vector<double> areas;
    for (int i = 1; i < num; ++i) {
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(i, cv::CC_STAT_TOP);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        const bool touchesBorder = (x <= 0) || (y <= 0) ||
                                   (x + w >= gray.cols) || (y + h >= gray.rows);
        if (touchesBorder) continue;

        keep.push_back(i);
        areas.push_back(static_cast<double>(stats.at<int>(i, cv::CC_STAT_AREA)));
    }

    if (keep.empty()) {
        std::cout << "未检测到任何目标" << std::endl;
        return 0;
    }

    std::vector<double> sorted = areas;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];
    const double minArea = std::max(200.0, 0.3 * median);

    // ---------- 步骤 7: 计数并绘制结果 ----------
    cv::Mat canvas = color.clone();
    int count = 0;
    for (size_t k = 0; k < keep.size(); ++k) {
        const int i = keep[k];
        if (areas[k] < minArea) continue;

        ++count;
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(i, cv::CC_STAT_TOP);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // centroids 是 N x 2 的 CV_64F 矩阵，必须用 at<double>(行, 列) 读取；
        // 用 at<cv::Point2d>(i) 会因 elemSize 不匹配触发断言直接中止
        const double cx = centroids.at<double>(i, 0);
        const double cy = centroids.at<double>(i, 1);

        cv::rectangle(canvas, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
        cv::circle(canvas, cv::Point(cvRound(cx), cvRound(cy)), 4, cv::Scalar(0, 0, 255), -1);
        cv::putText(canvas, std::to_string(count),
                    cv::Point(cvRound(cx) - 10, cvRound(cy) + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);

        std::cout << "硬币 #" << count << "  面积=" << static_cast<int>(areas[k])
                  << "  外接框=(" << x << "," << y << "," << w << "," << h << ")"
                  << "  质心=(" << cvRound(cx) << "," << cvRound(cy) << ")" << std::endl;
    }

    // 左上角写总数
    cv::putText(canvas, "count = " + std::to_string(count), cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 3);
    saveStep(outDir, "step7_result.png", canvas);

    std::cout << "----" << std::endl;
    std::cout << "Triangle 阈值 = " << th << "，面积下限 = " << static_cast<int>(minArea) << std::endl;
    std::cout << "硬币总数 = " << count << std::endl;
    return 0;
}
