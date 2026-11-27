# 练习3：Canny 边缘检测 + 霍夫直线检测

## 算法

1. 读入原始彩色图
2. 图像过小时等比放大（长边不足 600 时放大到 600，否则 Canny/霍夫的核与长度阈值失去意义）
3. 彩色转灰度
4. 高斯平滑 5×5，抑制噪点（Canny 对噪声极敏感）
5. **Canny 边缘检测**：双阈值由中位数自适应给出 `[0.66·med, 1.33·med]`；高阈值以上必留、低阈值以下必丢，中间的弱边缘只有与强边缘相连才保留（边缘连接 + 非极大值抑制）
6. **概率霍夫变换 `HoughLinesP`**：输出线段端点，可看到每段线的实际长度与角度
7. **标准霍夫变换 `HoughLines`**：输出极坐标 `(rho, theta)` 的整条直线，按累加器票数降序排列
8. 取票数最高的两条（跳过几乎重合的重复线）作为主干直线，用黄色画在原图上，并标注方向角与两线夹角

所有距离类参数按图像对角线长度归一化，换分辨率无需改参数：

| 参数 | 取值 |
|---|---|
| HoughLinesP 累加器阈值 | `0.05 × 对角线` |
| 线段最短长度 | `0.10 × 对角线` |
| 允许断裂间隙 | `0.02 × 对角线` |
| HoughLines 累加器阈值 | `0.10 × 对角线` |

## 文件

- `line_detect.cpp`：源程序（OpenCV C++）
- `image.jpg`：原始图片
- `step1_src.png`：原始彩色图
- `step2_resize.png`：等比放大后的处理图（长边 600）
- `step3_gray.png`：灰度图
- `step4_blur.png`：高斯平滑
- `step5_canny.png`：Canny 边缘图（对应练习截图左窗口）
- `step6_hough_segments.png`：HoughLinesP 线段（绿色）
- `step7_hough_lines.png`：HoughLines 检测到的全部直线（蓝色细线）
- `step8_result.png`：票数最高的两条主干直线（黄色）+ 角度标注（对应练习截图右窗口）

## 编译运行

```
cl /EHsc /std:c++17 /I "D:\OpenCV\opencv\build\include" line_detect.cpp ^
   /link /LIBPATH:"D:\OpenCV\opencv\build\x64\vc16\lib" opencv_world4120.lib
line_detect.exe image.jpg
```

## 结果

- 原图 108×48 → 放大 5.56 倍后处理尺寸 600×267
- Canny 双阈值 [66, 133]，边缘像素 2178
- HoughLinesP 检出 10 条线段，HoughLines 检出 15 条直线
- 主干两条直线方向角 6.0° / 7.0°，夹角 1.0°（即零件/传送带的上、下边界，互相平行）

> 说明：`image.jpg` 是零件的紧凑裁剪图（108×48），不含传送带本体，因此检测到的两条主干直线是零件自身的长边，而非练习截图里的传送带上下边界；把练习用的原始大图作为参数传入即可得到同样效果的传送带边界线，代码无需修改。
