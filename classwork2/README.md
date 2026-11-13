# 课堂练习：硬币计数

## 算法

基于**灰度阈值 + 连通域分析**的计数方法（硬币为亮目标，暗桌面为背景）：

1. 彩色转灰度
2. 高斯平滑，抑制桌面纹理与噪声
3. Triangle（三角形法）自动阈值二值化，硬币为白色前景
4. 形态学闭运算补缺口、开运算去噪点
5. 连通域标记：一枚硬币对应一个连通域
6. 剔除贴图像边框的连通域，并按中位面积自适应设定下限滤除碎屑
7. 统计剩余连通域个数即为硬币数量

说明：本图直方图为"大块暗背景 + 少量亮硬币"的强偏斜分布，Otsu 会把阈值抬到 133，导致偏暗的那枚硬币被切开；改用 Triangle 阈值（89）后 10 枚硬币均完整分割。

## 文件

- `coin_count.cpp`：源程序（OpenCV C++）
- `origin.png`：原始图片
- `step1_color.png` ~ `step7_result.png`：各处理步骤的中间结果

## 编译运行

```
cl /EHsc /std:c++17 /I "D:\OpenCV\opencv\build\include" coin_count.cpp ^
   /link /LIBPATH:"D:\OpenCV\opencv\build\x64\vc16\lib" opencv_world4120.lib
coin_count.exe origin.png
```

## 结果

检测到硬币 **10** 枚，与图中实际数量一致。
