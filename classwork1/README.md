# 课堂练习：回形针计数

## 算法

基于**灰度阈值 + 连通域分析**的计数方法：

1. 彩色转灰度（只保留亮度信息）
2. 高斯平滑，抑制桌面纹理与噪声
3. Otsu 自动阈值 + 反相二值化（回形针为暗目标，取反成白色前景）
4. 形态学闭运算补断线、开运算去噪点
5. 连通域标记：一根回形针的金属丝是闭合环，天然对应一个连通域
6. 剔除贴图像边框的连通域，并按中位面积自适应设定下限滤除碎屑
7. 统计剩余连通域个数即为回形针数量

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

检测到回形针 **13** 个，与图中实际数量一致。
