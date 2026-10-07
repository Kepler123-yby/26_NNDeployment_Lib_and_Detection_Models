#pragma once

#include "network_deployment_lib.hpp"

#include <opencv2/core.hpp>

#include <vector>

// ==================== 后处理模块（OpenCV CUDA 实现） ====================
// 在 GPU 上并行解码网络输出中的候选框，CPU 只负责 NMS 与最终结果组装。
// 仅在构建时启用 NNDEPLOYMENT_WITH_OPENCV_CUDA 时参与编译。

namespace MPT
{
// 装甲板 / 雷达候选解码结果，字段与 CPU 后处理中的临时容器保持一致。
struct CudaArmorCandidates
{
    std::vector<int> class_ids;         // 装甲板类别ID
    std::vector<float> confidences;     // 检测置信度（用于NMS）
    std::vector<int> sizes;             // 装甲板尺寸标志（0=小装甲板，1=大装甲板）
    std::vector<int> colors;            // 装甲板颜色类别（0=蓝，1=红，2=白，3=紫）
    std::vector<cv::Point2d> keypoints; // 每个候选 4 个关键点，按点顺序连续排列
    std::vector<cv::Rect> boxes;        // 边界框（用于OpenCV NMS算法）
};

// 神符候选解码结果。
struct CudaRuneCandidates
{
    std::vector<int> class_ids;         // 符叶类别ID
    std::vector<float> confidences;     // 检测置信度
    std::vector<cv::Point2d> keypoints; // 每个候选 5 个关键点，按点顺序连续排列
};

// V5 步兵模型解码：输出为 候选 × 特征（25200*22）。
CudaArmorCandidates cudaDecodeArmorV5(const float *input_ptr, const InferParam &infer_param,
                                      float confidence_threshold, int my_color);

// V8 / V8_21 步兵模型解码：输出为 特征 × 候选，point_stride 为关键点步长（V8=3，V8_21=2）。
CudaArmorCandidates cudaDecodeArmorV8(const float *input_ptr, const InferParam &infer_param,
                                      float confidence_threshold, int my_color, int point_stride);

// 雷达模型解码：输出为 特征 × 候选，使用 xywh 作为 NMS 边界框。
CudaArmorCandidates cudaDecodeLidar(const float *input_ptr, const InferParam &infer_param,
                                    float confidence_threshold);

// 神符模型解码：输出为 特征 × 候选。
CudaRuneCandidates cudaDecodeRune(const float *input_ptr, const InferParam &infer_param,
                                  float confidence_threshold);
} // namespace MPT
