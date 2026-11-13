# 课堂练习：回形针计数

## 算法

基于**灰度阈值 + 连通域分析 + 形状过滤**的计数方法：

1. 彩色转灰度（只保留亮度信息）
2. 高斯平滑，抑制桌面纹理与噪声
3. Otsu 自动阈值 + 反相二值化（回形针为暗目标，取反成白色前景）
4. 形态学闭运算补断线、开运算去噪点
5. 连通域标记：一根回形针的金属丝是闭合环，天然对应一个连通域
6. 三重过滤：
   - 剔除贴图像边框的连通域（照片暗边框会被当成大目标）
   - **长宽比过滤**：最小外接矩形长宽比需在 `[2.0, 4.5]` 内。回形针细长，实测 13 个目标稳定在 3.08~3.40；圆点、块状污渍接近 1:1，划痕远大于 4.5，都会被剔除
   - 面积过滤：以候选目标中位面积的 30% 为下限，滤除碎屑噪点
7. 统计剩余连通域个数即为回形针数量

长宽比用 `minAreaRect`（最小外接矩形）而不是 `boundingRect`：后者是水平矩形，回形针一斜长宽比就变成 1:1，无法作为形状特征。

## 文件

- `paperclip_count.cpp`：源程序（OpenCV C++）
- `origin.png`：原始图片
- `step1_color.png` ~ `step7_result.png`：各处理步骤的中间结果

## 编译运行

```
cl /EHsc /std:c++17 /I "D:\OpenCV\opencv\build\include" paperclip_count.cpp ^
   /link /LIBPATH:"D:\OpenCV\opencv\build\x64\vc16\lib" opencv_world4120.lib
paperclip_count.exe origin.png
```

（也可直接用 VS Code 任务 `build-opencv` / `run-opencv`）

## 结果

检测到回形针 **13** 个，与图中实际数量一致；13 个目标的长宽比为 3.08~3.40，全部通过形状过滤。
