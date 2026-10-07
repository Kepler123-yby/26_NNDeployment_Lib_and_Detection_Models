#include "network_inference.hpp"
#include "network_preprocess_cuda.hpp"

#if NNDEPLOYMENT_WITH_OPENCV_CUDA

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>

// ==================== 预处理模块（OpenCV CUDA 实现） ====================
// 使用 OpenCV CUDA 在 GPU 上完成缩放与 padding，再下载回 host 侧的预处理缓冲区。
// 该文件仅在构建时启用 NNDEPLOYMENT_WITH_OPENCV_CUDA 时参与编译。

bool YOLOModel::InferenceEngine::cudaResizeAndPad(const cv::Mat &origin_image,
                                                  int target_width, int target_height,
                                                  int new_width, int new_height,
                                                  int pad_x, int pad_y,
                                                  cv::Mat &output_buffer)
{
    if (origin_image.empty() || target_width <= 0 || target_height <= 0)
        return false;
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
        return false;

    try
    {
        cv::cuda::GpuMat gpu_origin;
        gpu_origin.upload(origin_image);

        cv::cuda::GpuMat gpu_output;
        if (new_height == target_height && new_width == target_width)
        {
            // 长宽比一致时直接缩放，无需 padding。
            cv::cuda::resize(gpu_origin, gpu_output,
                             cv::Size(target_width, target_height),
                             0, 0, cv::INTER_LINEAR);
        }
        else
        {
            cv::cuda::GpuMat gpu_resized;
            cv::cuda::resize(gpu_origin, gpu_resized,
                             cv::Size(new_width, new_height),
                             0, 0, cv::INTER_LINEAR);
            cv::cuda::copyMakeBorder(gpu_resized, gpu_output,
                                     pad_y, target_height - new_height - pad_y,
                                     pad_x, target_width - new_width - pad_x,
                                     cv::BORDER_CONSTANT, cv::Scalar(124, 124, 124));
        }

        gpu_output.download(output_buffer);
        return !output_buffer.empty();
    }
    catch (const cv::Exception &error)
    {
        std::cerr << "OpenCV CUDA 预处理失败，已回退到 CPU 路径：" << error.what() << std::endl;
        return false;
    }
}

namespace MPT
{
namespace
{
// 把已经是目标尺寸的 BGR u8 图（HWC 交织）转换为 TensorRT 输入所需的
// RGB float NCHW（除以 255），直接写入 device_input。
// 注意：GpuMat 的每行按硬件对齐，src_step 不一定等于 cols*3，必须按行步长取像素。
__global__ void letterboxBgrToRgbNchwKernel(const unsigned char *src, int src_step,
                                            int rows, int cols,
                                            float *dst, float scale)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (index >= total)
        return;

    const int y = index / cols;
    const int x = index % cols;
    const unsigned char *pixel =
        src + static_cast<std::size_t>(y) * src_step + static_cast<std::size_t>(x) * 3;

    const float b = static_cast<float>(pixel[0]) * scale;
    const float g = static_cast<float>(pixel[1]) * scale;
    const float r = static_cast<float>(pixel[2]) * scale;

    dst[index] = r;                 // plane 0: R
    dst[total + index] = g;         // plane 1: G
    dst[2 * total + index] = b;     // plane 2: B
}
} // namespace

bool cudaFuseTensorRTInput(const cv::Mat &origin_image,
                           int target_width, int target_height,
                           float *device_input,
                           cudaStream_t stream,
                           InferParam &infer_param)
{
    if (origin_image.empty() || device_input == nullptr ||
        target_width <= 0 || target_height <= 0)
        return false;
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
        return false;

    try
    {
        // 把 TensorRT 的 CUDA 流包装成 cv::cuda::Stream，保证后续 resize/padding 与
        // kernel、以及 TensorRT 的 enqueueV3 在同一流上有序。
        cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

        infer_param.origin_width = origin_image.cols;
        infer_param.origin_height = origin_image.rows;

        const float scale = std::min(static_cast<float>(target_height) / origin_image.rows,
                                     static_cast<float>(target_width) / origin_image.cols);
        const int new_width = std::min(target_width, static_cast<int>(std::round(origin_image.cols * scale)));
        const int new_height = std::min(target_height, static_cast<int>(std::round(origin_image.rows * scale)));
        if (new_width <= 0 || new_height <= 0)
            return false;

        infer_param.scale = scale;
        infer_param.pad_x = std::max(0, (target_width - new_width) / 2);
        infer_param.pad_y = std::max(0, (target_height - new_height) / 2);

        cv::cuda::GpuMat gpu_origin;
        gpu_origin.upload(origin_image, cv_stream);

        cv::cuda::GpuMat gpu_bgr;
        if (new_width == target_width && new_height == target_height)
        {
            cv::cuda::resize(gpu_origin, gpu_bgr, cv::Size(target_width, target_height),
                             0, 0, cv::INTER_LINEAR, cv_stream);
        }
        else
        {
            cv::cuda::GpuMat gpu_resized;
            cv::cuda::resize(gpu_origin, gpu_resized, cv::Size(new_width, new_height),
                             0, 0, cv::INTER_LINEAR, cv_stream);
            cv::cuda::copyMakeBorder(gpu_resized, gpu_bgr,
                                     infer_param.pad_y, target_height - new_height - infer_param.pad_y,
                                     infer_param.pad_x, target_width - new_width - infer_param.pad_x,
                                     cv::BORDER_CONSTANT, cv::Scalar(124, 124, 124), cv_stream);
        }

        const int total = target_width * target_height;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        letterboxBgrToRgbNchwKernel<<<blocks, threads, 0, stream>>>(
            gpu_bgr.ptr<unsigned char>(), static_cast<int>(gpu_bgr.step),
            target_height, target_width, device_input, 1.0f / 255.0f);

        return true;
    }
    catch (const cv::Exception &error)
    {
        std::cerr << "OpenCV CUDA 融合预处理失败，已回退到 CPU + blobFromImage：" << error.what() << std::endl;
        return false;
    }
}
} // namespace MPT

#endif
