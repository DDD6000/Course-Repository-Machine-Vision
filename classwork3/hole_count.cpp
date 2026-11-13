// 圆孔计数 (counting complete circular holes)
// 目标: 只统计 4 个完整圆形孔；不完整的孔(被图像边界切断)、非圆形的孔都不计入
//
// 处理流程:
//   裁掉扫描黑边 -> 灰度 -> 高斯平滑 -> Otsu 反相二值化 -> 形态学 -> 连通域标记
//   -> 过滤(贴边 / 圆度 / 面积) -> 计数
//
// 每一步的中间结果都会保存成图片，方便观察算法过程
// 编译: 调用 vcvars64.bat 后用 cl 编译，链接 opencv_world4120.lib
// 运行: hole_count.exe [图片路径]   默认 origin.png

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

// 圆度 = 4*pi*A / P^2，正圆为 1，越不圆越小。
// 需要对单个标号取最外轮廓来求周长，所以先抠出该标号的掩膜再做 findContours。
static double circularityOf(const cv::Mat& labels, int id) {
    cv::Mat mask = (labels == id);  // 0/255 的单通道掩膜
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return 0.0;

    // 取面积最大的外轮廓
    size_t best = 0;
    double bestArea = -1.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double a = cv::contourArea(contours[i]);
        if (a > bestArea) {
            bestArea = a;
            best = i;
        }
    }
    const double perimeter = cv::arcLength(contours[best], true);
    if (perimeter <= 0.0) return 0.0;
    return 4.0 * CV_PI * bestArea / (perimeter * perimeter);
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

    // ---------- 步骤 1: 裁掉四周扫描黑边 ----------
    // 图像边缘有一圈纯黑扫描边框，二值化后会连成一个大环，还会把贴边的孔洞"粘"进去，
    // 先裁掉 margin 像素，后面的"贴边即剔除"规则才能准确工作。
    const int margin = 8;
    if (color.cols <= 2 * margin || color.rows <= 2 * margin) {
        std::cerr << "图像尺寸过小" << std::endl;
        return -1;
    }
    cv::Mat roi = color(cv::Rect(margin, margin,
                                 color.cols - 2 * margin, color.rows - 2 * margin)).clone();
    saveStep(outDir, "step2_crop.png", roi);

    // ---------- 步骤 2: 彩色转灰度 ----------
    // 孔洞比金属表面暗，只需要亮度信息
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    saveStep(outDir, "step3_gray.png", gray);

    // ---------- 步骤 3: 高斯平滑 ----------
    // 抑制金属表面纹理、反光和噪声
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    saveStep(outDir, "step4_blur.png", blur);

    // ---------- 步骤 4: Otsu 自动阈值二值化 + 反相 ----------
    // 孔洞是暗目标，反相(THRESH_BINARY_INV)后孔洞变成白色前景
    cv::Mat binary;
    const double otsuTh = cv::threshold(blur, binary, 0, 255,
                                        cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    saveStep(outDir, "step5_binary.png", binary);

    // ---------- 步骤 5: 形态学处理 ----------
    // 闭运算补细小缺口(避免一个孔被拆成几块)，开运算去掉孤立噪点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    saveStep(outDir, "step6_morph.png", binary);

    // ---------- 步骤 6: 连通域标记 ----------
    // 每个孔洞是一块独立的暗区，8 邻域标记后一个孔 = 一个连通域
    cv::Mat labels, stats, centroids;
    const int num = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

    // 连通域伪彩色图: 每个标号一种颜色，直观检查有没有粘连或碎裂
    cv::Mat labelsU8, labelsVis;
    labels.convertTo(labelsU8, CV_8U, 12);
    cv::applyColorMap(labelsU8, labelsVis, cv::COLORMAP_JET);
    labelsVis.setTo(cv::Vec3b(0, 0, 0), labels == 0);  // 背景涂黑
    saveStep(outDir, "step7_labels.png", labelsVis);

    // ---------- 步骤 7: 三重过滤，只留"完整的圆" ----------
    struct Candidate {
        int id;
        double area;
        double circ;
        cv::Rect box;
    };
    std::vector<Candidate> candidates;  // 通过贴边 + 圆度筛选
    std::vector<cv::Rect> rejected;     // 被剔除的，画到结果图里做对照

    const double kMinCircularity = 0.75;  // 圆度下限，正圆为 1
    for (int i = 1; i < num; ++i) {
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(i, cv::CC_STAT_TOP);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const cv::Rect box(x, y, w, h);
        const double area = static_cast<double>(stats.at<int>(i, cv::CC_STAT_AREA));

        // (a) 贴边的不完整目标: 被图像边界切断的大孔、左侧不规则孔都贴边
        if (x <= 0 || y <= 0 || x + w >= gray.cols || y + h >= gray.rows) {
            rejected.push_back(box);
            continue;
        }
        // (b) 圆度不够的不是圆孔(不规则暗区、细长划痕等)
        const double circ = circularityOf(labels, i);
        if (circ < kMinCircularity) {
            rejected.push_back(box);
            continue;
        }
        candidates.push_back({i, area, circ, box});
    }

    // (c) 面积过滤: 以圆形候选面积的中位数为基准去掉碎屑噪点
    double minArea = 0.0;
    if (!candidates.empty()) {
        std::vector<double> areas;
        for (const Candidate& c : candidates) areas.push_back(c.area);
        std::sort(areas.begin(), areas.end());
        const double median = areas[areas.size() / 2];
        minArea = std::max(100.0, 0.25 * median);
    }

    // ---------- 步骤 8: 计数并绘制结果 ----------
    cv::Mat canvas = roi.clone();

    // 先画被剔除的目标(青色框)，便于和保留的对比
    for (const cv::Rect& r : rejected) {
        cv::rectangle(canvas, r, cv::Scalar(255, 255, 0), 1);
    }

    int count = 0;
    for (const Candidate& c : candidates) {
        if (c.area < minArea) {
            cv::rectangle(canvas, c.box, cv::Scalar(255, 255, 0), 1);
            continue;
        }
        ++count;

        // centroids 是 N x 2 的 CV_64F 矩阵，必须用 at<double>(行, 列) 读取；
        // 用 at<cv::Point2d>(i) 会因 elemSize 不匹配触发断言直接中止
        const double cx = centroids.at<double>(c.id, 0);
        const double cy = centroids.at<double>(c.id, 1);

        cv::rectangle(canvas, c.box, cv::Scalar(0, 255, 0), 2);
        cv::circle(canvas, cv::Point(cvRound(cx), cvRound(cy)), 4, cv::Scalar(0, 0, 255), -1);
        cv::putText(canvas, std::to_string(count),
                    cv::Point(cvRound(cx) - 10, cvRound(cy) + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);

        std::cout << "圆孔 #" << count << "  面积=" << static_cast<int>(c.area)
                  << "  圆度=" << cv::format("%.2f", c.circ)
                  << "  外接框=(" << c.box.x << "," << c.box.y << ","
                  << c.box.width << "," << c.box.height << ")"
                  << "  质心=(" << cvRound(cx) << "," << cvRound(cy) << ")" << std::endl;
    }

    cv::putText(canvas, "count = " + std::to_string(count), cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 3);
    saveStep(outDir, "step8_result.png", canvas);

    std::cout << "----" << std::endl;
    std::cout << "Otsu 阈值 = " << otsuTh << "，圆度下限 = " << kMinCircularity
              << "，面积下限 = " << static_cast<int>(minArea) << std::endl;
    std::cout << "完整圆孔数量 = " << count << std::endl;
    return 0;
}
