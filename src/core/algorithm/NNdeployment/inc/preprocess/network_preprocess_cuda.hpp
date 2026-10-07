#pragma once

#include "network_deployment_lib.hpp"

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>

#ifndef NNDEPLOYMENT_WITH_OPENCV_CUDA
#define NNDEPLOYMENT_WITH_OPENCV_CUDA 0
#endif

#if NNDEPLOYMENT_WITH_OPENCV_CUDA
namespace MPT
{
// 在 GPU 上一次性完成 letterbox（resize + padding）+ BGR→RGB + /255 + HWC→CHW，
// 直接把结果写进 TensorRT 的输入显存（1×3×H×W 连续的 float），并回填坐标还原参数
// （origin_width/height、scale、pad_x/pad_y）。这样就不需要在 CPU 上跑
// cv::dnn::blobFromImage。
//
// 返回 false 表示 CUDA 不可用或执行失败，调用方应回退到 CPU 预处理 + blobFromImage。
// 注意：只更新 infer_param 的坐标相关字段，不改动 out_tensor_rows/cols。
bool cudaFuseTensorRTInput(const cv::Mat &origin_image,
                           int target_width, int target_height,
                           float *device_input,
                           cudaStream_t stream,
                           InferParam &infer_param);
} // namespace MPT
#endif
