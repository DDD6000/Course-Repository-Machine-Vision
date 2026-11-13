// 回形针计数 (paperclip counting)
// 处理流程: 彩色->灰度 -> 高斯平滑 -> Otsu二值化(反相) -> 形态学 -> 连通域 -> 面积/边框过滤 -> 计数
// 每一步的中间结果都会保存成图片，方便观察算法过程
//
// 编译: 调用 vcvars64.bat 后用 cl 编译，链接 opencv_world4120.lib
// 运行: paperclip_count.exe [图片路径]   默认 origin.png

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
    // 只关心"暗的回形针 vs 亮的桌面"这个亮度差异，颜色信息是噪声，去掉可减少计算量
    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    saveStep(outDir, "step2_gray.png", gray);

    // ---------- 步骤 2: 高斯平滑 ----------
    // 抑制桌面纹理、胶片颗粒和压缩噪声，避免二值化后出现大量毛刺噪点
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    saveStep(outDir, "step3_blur.png", blur);

    // ---------- 步骤 3: Otsu 自动阈值二值化 + 反相 ----------
    // Otsu 通过最大化类间方差自动找阈值(这里约 118)；
    // 回形针比桌面暗，反相(THRESH_BINARY_INV)后回形针变白色前景(255)，背景变黑
    cv::Mat binary;
    const double otsuTh = cv::threshold(blur, binary, 0, 255,
                                        cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    saveStep(outDir, "step4_binary.png", binary);

    // ---------- 步骤 4: 形态学处理 ----------
    // 闭运算(先膨胀后腐蚀)：把金属丝上因反光/磨损造成的细小断裂连起来，
    //   否则一根回形针可能被拆成两段，计数会偏多；
    // 开运算(先腐蚀后膨胀)：去掉孤立的小噪点。
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    saveStep(outDir, "step5_morph.png", binary);

    // ---------- 步骤 5: 连通域标记 ----------
    // 每根回形针的金属丝本身就是一个闭合环，所以"一根回形针 = 一个连通域"，
    // 不需要先填洞再腐蚀分离。8 邻域连通标记，同时得到每个连通域的面积、外接矩形、质心。
    cv::Mat labels, stats, centroids;
    const int num = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

    // 连通域伪彩色图：给每个标号一个不同颜色，直观看出分割是否把一根回形针拆成了两块
    cv::Mat labelsU8, labelsVis;
    labels.convertTo(labelsU8, CV_8U, 12);  // 标号拉开放大，颜色区分更明显
    cv::applyColorMap(labelsU8, labelsVis, cv::COLORMAP_JET);
    labelsVis.setTo(cv::Vec3b(0, 0, 0), labels == 0);  // 背景涂黑
    saveStep(outDir, "step6_labels.png", labelsVis);

    // ---------- 步骤 6: 过滤 ----------
    // (a) 剔除"贴到图像边框"的连通域：照片暗边框/扫描黑边会被当成一个大目标
    // (b) 面积过滤：以剩余目标的中位面积自适应定下限，去掉碎屑噪点
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

        // 注意: centroids 是 N x 2 的 CV_64F 矩阵，用 at<double>(行, 列) 读取；
        // 不能用 at<cv::Point2d>(i)，OpenCV 4.12 会因 elemSize 不匹配触发断言直接中止
        const double cx = centroids.at<double>(i, 0);
        const double cy = centroids.at<double>(i, 1);

        cv::rectangle(canvas, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
        cv::circle(canvas, cv::Point(cvRound(cx), cvRound(cy)), 4, cv::Scalar(0, 0, 255), -1);
        cv::putText(canvas, std::to_string(count),
                    cv::Point(cvRound(cx) + 6, cvRound(cy) - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);

        std::cout << "回形针 #" << count << "  面积=" << static_cast<int>(areas[k])
                  << "  外接框=(" << x << "," << y << "," << w << "," << h << ")"
                  << "  质心=(" << cvRound(cx) << "," << cvRound(cy) << ")" << std::endl;
    }

    // 左上角写总数
    cv::putText(canvas, "count = " + std::to_string(count), cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
    saveStep(outDir, "step7_result.png", canvas);

    std::cout << "----" << std::endl;
    std::cout << "Otsu 阈值 = " << otsuTh << "，面积下限 = " << static_cast<int>(minArea) << std::endl;
    std::cout << "回形针总数 = " << count << std::endl;
    return 0;
}
