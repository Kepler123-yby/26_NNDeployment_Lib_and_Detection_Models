#include "network_postprocess_cuda.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// ==================== 后处理模块（OpenCV CUDA 实现） ====================
// 使用 OpenCV CUDA 的 GpuMat 承载网络输出，并用自定义 kernel 在 GPU 上并行解码候选。
// NMS 与最终结果组装仍在 CPU 上完成，保持与纯 CPU 路径完全一致的语义。
// 该文件仅在构建时启用 NNDEPLOYMENT_WITH_OPENCV_CUDA 时参与编译。

namespace MPT
{
namespace
{
constexpr int kBlockSize = 256;

// 根据候选数量计算 kernel 的网格大小。
int gridSize(int total)
{
    return (total + kBlockSize - 1) / kBlockSize;
}

// 与 CPU 路径一致的 sigmoid。
__device__ __forceinline__ float sigmoidDevice(float x)
{
    return x >= 0.f ? 1.f / (1.f + __expf(-x)) : __expf(x) / (1.f + __expf(x));
}

// V5：输出为 候选 × 特征。关键点在特征 0..7，置信度在 8，颜色在 9..12，类别在 13..21。
__global__ void decodeArmorV5Kernel(const float *data, int num_candidates, int num_features,
                                    float raw_threshold, int my_color,
                                    int *count, int *class_ids, int *colors,
                                    float *confidences, float *keypoints)
{
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= num_candidates)
        return;

    const float *row = data + static_cast<std::size_t>(candidate) * num_features;
    const float raw_confidence = row[8];
    if (raw_confidence < raw_threshold)
        return;

    int class_id = 0;
    float class_score = row[13];
    for (int i = 1; i < 9; ++i)
    {
        const float score = row[13 + i];
        if (score > class_score)
        {
            class_score = score;
            class_id = i;
        }
    }

    int color_id = 0;
    float color_score = row[9];
    for (int i = 1; i < 4; ++i)
    {
        const float score = row[9 + i];
        if (score > color_score)
        {
            color_score = score;
            color_id = i;
        }
    }

    if (color_id == 3)
        return;
    if (color_id == 0 && my_color == 1)
        return;
    if (color_id == 1 && my_color == 0)
        return;

    const int slot = atomicAdd(count, 1);
    class_ids[slot] = class_id;
    colors[slot] = color_id;
    confidences[slot] = sigmoidDevice(raw_confidence);
#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        keypoints[slot * 8 + p * 2 + 0] = row[p * 2 + 0];
        keypoints[slot * 8 + p * 2 + 1] = row[p * 2 + 1];
    }
}

// V8 / V8_21：输出为 特征 × 候选。颜色在 0..3，类别在 4..12，关键点从 13 开始按 PointStride 排布。
template <int PointStride>
__global__ void decodeArmorV8Kernel(const float *data, int num_candidates, int feature_stride,
                                    float confidence_threshold, int my_color,
                                    int *count, int *class_ids, int *colors,
                                    float *confidences, float *keypoints)
{
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= num_candidates)
        return;

    auto feature = [&](int index) -> float
    {
        return data[static_cast<std::size_t>(index) * feature_stride + candidate];
    };

    int class_id = 0;
    float class_score = feature(4);
    for (int i = 1; i < 9; ++i)
    {
        const float score = feature(4 + i);
        if (score > class_score)
        {
            class_score = score;
            class_id = i;
        }
    }
    if (class_score < confidence_threshold)
        return;

    int color_id = 0;
    float color_score = feature(0);
    for (int i = 1; i < 4; ++i)
    {
        const float score = feature(i);
        if (score > color_score)
        {
            color_score = score;
            color_id = i;
        }
    }

    if (color_id == 3)
        return;
    if (color_id == 0 && my_color == 1)
        return;
    if (color_id == 1 && my_color == 0)
        return;

    const int slot = atomicAdd(count, 1);
    class_ids[slot] = class_id;
    colors[slot] = color_id;
    confidences[slot] = class_score;
#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        keypoints[slot * 8 + p * 2 + 0] = feature(13 + p * PointStride + 0);
        keypoints[slot * 8 + p * 2 + 1] = feature(13 + p * PointStride + 1);
    }
}

// 雷达：输出为 特征 × 候选。xywh 在 0..3，类别在 4..13，关键点从 14 开始按步长 3 排布。
__global__ void decodeLidarKernel(const float *data, int num_candidates, int feature_stride,
                                  float confidence_threshold,
                                  int *count, int *class_ids,
                                  float *confidences, float *boxes, float *keypoints)
{
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= num_candidates)
        return;

    auto feature = [&](int index) -> float
    {
        return data[static_cast<std::size_t>(index) * feature_stride + candidate];
    };

    int class_id = 0;
    float class_score = feature(4);
    for (int i = 1; i < 10; ++i)
    {
        const float score = feature(4 + i);
        if (score > class_score)
        {
            class_score = score;
            class_id = i;
        }
    }
    if (class_score < confidence_threshold)
        return;

    const int slot = atomicAdd(count, 1);
    class_ids[slot] = class_id;
    confidences[slot] = class_score;
    boxes[slot * 4 + 0] = feature(0);
    boxes[slot * 4 + 1] = feature(1);
    boxes[slot * 4 + 2] = feature(2);
    boxes[slot * 4 + 3] = feature(3);
#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        keypoints[slot * 8 + p * 2 + 0] = feature(14 + p * 3 + 0);
        keypoints[slot * 8 + p * 2 + 1] = feature(14 + p * 3 + 1);
    }
}

// 大符：输出为 特征 × 候选。类别在 0..2，关键点从 3 开始按步长 3 排布（第三个数为关键点置信度）。
__global__ void decodeRuneKernel(const float *data, int num_candidates, int feature_stride,
                                 float confidence_threshold,
                                 int *count, int *class_ids,
                                 float *confidences, float *keypoints)
{
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= num_candidates)
        return;

    auto feature = [&](int index) -> float
    {
        return data[static_cast<std::size_t>(index) * feature_stride + candidate];
    };

    int class_id = 0;
    float class_score = feature(0);
    for (int i = 1; i < 3; ++i)
    {
        const float score = feature(i);
        if (score > class_score)
        {
            class_score = score;
            class_id = i;
        }
    }
    if (class_score < confidence_threshold)
        return;

    // 至少 3 个关键点置信度超过阈值才保留。
    int valid_keypoints = 0;
#pragma unroll
    for (int p = 0; p < 5; ++p)
    {
        if (feature(3 + p * 3 + 2) > 0.8f)
            ++valid_keypoints;
    }
    if (valid_keypoints < 3)
        return;

    const int slot = atomicAdd(count, 1);
    class_ids[slot] = class_id;
    confidences[slot] = class_score;
#pragma unroll
    for (int p = 0; p < 5; ++p)
    {
        keypoints[slot * 10 + p * 2 + 0] = feature(3 + p * 3 + 0);
        keypoints[slot * 10 + p * 2 + 1] = feature(3 + p * 3 + 1);
    }
}

// 复用同一线程的显存缓冲，避免每帧重复 cudaMalloc。
// 使用 new 创建并有意不释放，避免静态 GpuMat 在 CUDA 上下文销毁后再析构。
struct ArmorScratch
{
    cv::cuda::GpuMat input;
    cv::cuda::GpuMat class_ids;
    cv::cuda::GpuMat colors;
    cv::cuda::GpuMat confidences;
    cv::cuda::GpuMat keypoints;
    cv::cuda::GpuMat boxes;
    cv::cuda::GpuMat count;
};

ArmorScratch &armorScratch()
{
    static thread_local ArmorScratch *scratch = new ArmorScratch();
    return *scratch;
}

// 将 GPU 上的候选整理为与 CPU 后处理一致的容器（还原坐标、生成边界框）。
CudaArmorCandidates collectArmorCandidates(const ArmorScratch &scratch,
                                           const InferParam &infer_param,
                                           bool use_xywh_box)
{
    CudaArmorCandidates result;

    cv::Mat host_count;
    scratch.count.download(host_count);
    const int candidate_count = host_count.at<int>(0);
    if (candidate_count <= 0)
        return result;

    cv::Mat host_class_ids;
    cv::Mat host_colors;
    cv::Mat host_confidences;
    cv::Mat host_keypoints;
    cv::Mat host_boxes;
    scratch.class_ids.colRange(0, candidate_count).download(host_class_ids);
    scratch.colors.colRange(0, candidate_count).download(host_colors);
    scratch.confidences.colRange(0, candidate_count).download(host_confidences);
    scratch.keypoints.colRange(0, candidate_count * 8).download(host_keypoints);
    if (use_xywh_box)
        scratch.boxes.colRange(0, candidate_count * 4).download(host_boxes);

    const int *class_ptr = host_class_ids.ptr<int>();
    const int *color_ptr = host_colors.ptr<int>();
    const float *confidence_ptr = host_confidences.ptr<float>();
    const float *keypoint_ptr = host_keypoints.ptr<float>();
    const float *box_ptr = use_xywh_box ? host_boxes.ptr<float>() : nullptr;

    result.class_ids.reserve(candidate_count);
    result.confidences.reserve(candidate_count);
    result.sizes.reserve(candidate_count);
    result.colors.reserve(candidate_count);
    result.keypoints.reserve(static_cast<std::size_t>(candidate_count) * 4);
    result.boxes.reserve(candidate_count);

    for (int index = 0; index < candidate_count; ++index)
    {
        std::vector<cv::Point2f> points(4);
        for (int point = 0; point < 4; ++point)
        {
            const float x = keypoint_ptr[index * 8 + point * 2 + 0];
            const float y = keypoint_ptr[index * 8 + point * 2 + 1];
            points[point] = cv::Point2f((x - infer_param.pad_x) / infer_param.scale,
                                        (y - infer_param.pad_y) / infer_param.scale);
        }

        const int class_id = class_ptr[index];
        result.class_ids.push_back(class_id);
        result.confidences.push_back(confidence_ptr[index]);
        result.colors.push_back(color_ptr[index]);

        if (use_xywh_box)
        {
            // 雷达使用网络输出的 xywh 作为 NMS 边界框，还原方式与 CPU 路径一致。
            const float cx = box_ptr[index * 4 + 0];
            const float cy = box_ptr[index * 4 + 1];
            const float w = box_ptr[index * 4 + 2];
            const float h = box_ptr[index * 4 + 3];
            const int left = std::max(0.f, (((cx - 0.5f * w) - infer_param.pad_x) / infer_param.scale) + 0.5f);
            const int top = std::max(0.f, (((cy - 0.5f * h) - infer_param.pad_y) / infer_param.scale) + 0.5f);
            const int width = static_cast<int>(w / infer_param.scale + 0.5);
            const int height = static_cast<int>(h / infer_param.scale + 0.5);
            result.boxes.emplace_back(left, top, width, height);
            result.sizes.push_back(0); // 雷达装甲板不需要 size 属性
        }
        else
        {
            result.boxes.push_back(cv::boundingRect(points));
            result.sizes.push_back((class_id == 1 || class_id == 7) ? 1 : 0);
        }

        for (const cv::Point2f &point : points)
            result.keypoints.emplace_back(point.x, point.y);
    }

    return result;
}
} // namespace

CudaArmorCandidates cudaDecodeArmorV5(const float *input_ptr, const InferParam &infer_param,
                                      float confidence_threshold, int my_color,
                                      const void *device_input)
{
    const int num_candidates = infer_param.out_tensor_rows;
    const int num_features = infer_param.out_tensor_cols;

    ArmorScratch &scratch = armorScratch();
    const std::size_t total = static_cast<std::size_t>(num_candidates) * num_features;
    const float *data_ptr = nullptr;
    if (device_input != nullptr)
    {
        data_ptr = static_cast<const float *>(device_input);
    }
    else
    {
        scratch.input.create(1, static_cast<int>(total), CV_32F);
        cudaMemcpy(scratch.input.data, input_ptr, total * sizeof(float), cudaMemcpyHostToDevice);
        data_ptr = reinterpret_cast<const float *>(scratch.input.data);
    }

    scratch.class_ids.create(1, num_candidates, CV_32S);
    scratch.colors.create(1, num_candidates, CV_32S);
    scratch.confidences.create(1, num_candidates, CV_32F);
    scratch.keypoints.create(1, num_candidates * 8, CV_32F);
    scratch.count.create(1, 1, CV_32S);
    scratch.count.setTo(cv::Scalar(0));

    const float threshold = confidence_threshold;
    const float raw_threshold = static_cast<float>(std::log(threshold / (1.0f - threshold)));

    decodeArmorV5Kernel<<<gridSize(num_candidates), kBlockSize>>>(
        data_ptr, num_candidates, num_features,
        raw_threshold, my_color,
        reinterpret_cast<int *>(scratch.count.data),
        reinterpret_cast<int *>(scratch.class_ids.data),
        reinterpret_cast<int *>(scratch.colors.data),
        reinterpret_cast<float *>(scratch.confidences.data),
        reinterpret_cast<float *>(scratch.keypoints.data));
    cudaDeviceSynchronize();

    return collectArmorCandidates(scratch, infer_param, false);
}

CudaArmorCandidates cudaDecodeArmorV8(const float *input_ptr, const InferParam &infer_param,
                                      float confidence_threshold, int my_color, int point_stride,
                                      const void *device_input)
{
    const int num_candidates = infer_param.out_tensor_cols;
    const int num_features = infer_param.out_tensor_rows;

    ArmorScratch &scratch = armorScratch();
    const std::size_t total = static_cast<std::size_t>(num_candidates) * num_features;
    const float *data_ptr = nullptr;
    if (device_input != nullptr)
    {
        data_ptr = static_cast<const float *>(device_input);
    }
    else
    {
        scratch.input.create(1, static_cast<int>(total), CV_32F);
        cudaMemcpy(scratch.input.data, input_ptr, total * sizeof(float), cudaMemcpyHostToDevice);
        data_ptr = reinterpret_cast<const float *>(scratch.input.data);
    }

    scratch.class_ids.create(1, num_candidates, CV_32S);
    scratch.colors.create(1, num_candidates, CV_32S);
    scratch.confidences.create(1, num_candidates, CV_32F);
    scratch.keypoints.create(1, num_candidates * 8, CV_32F);
    scratch.count.create(1, 1, CV_32S);
    scratch.count.setTo(cv::Scalar(0));

    if (point_stride == 2)
    {
        decodeArmorV8Kernel<2><<<gridSize(num_candidates), kBlockSize>>>(
            data_ptr, num_candidates, num_candidates, confidence_threshold, my_color,
            reinterpret_cast<int *>(scratch.count.data),
            reinterpret_cast<int *>(scratch.class_ids.data),
            reinterpret_cast<int *>(scratch.colors.data),
            reinterpret_cast<float *>(scratch.confidences.data),
            reinterpret_cast<float *>(scratch.keypoints.data));
    }
    else
    {
        decodeArmorV8Kernel<3><<<gridSize(num_candidates), kBlockSize>>>(
            data_ptr, num_candidates, num_candidates, confidence_threshold, my_color,
            reinterpret_cast<int *>(scratch.count.data),
            reinterpret_cast<int *>(scratch.class_ids.data),
            reinterpret_cast<int *>(scratch.colors.data),
            reinterpret_cast<float *>(scratch.confidences.data),
            reinterpret_cast<float *>(scratch.keypoints.data));
    }
    cudaDeviceSynchronize();

    return collectArmorCandidates(scratch, infer_param, false);
}

CudaArmorCandidates cudaDecodeLidar(const float *input_ptr, const InferParam &infer_param,
                                    float confidence_threshold,
                                    const void *device_input)
{
    const int num_candidates = infer_param.out_tensor_cols;
    const int num_features = infer_param.out_tensor_rows;

    ArmorScratch &scratch = armorScratch();
    const std::size_t total = static_cast<std::size_t>(num_candidates) * num_features;
    const float *data_ptr = nullptr;
    if (device_input != nullptr)
    {
        data_ptr = static_cast<const float *>(device_input);
    }
    else
    {
        scratch.input.create(1, static_cast<int>(total), CV_32F);
        cudaMemcpy(scratch.input.data, input_ptr, total * sizeof(float), cudaMemcpyHostToDevice);
        data_ptr = reinterpret_cast<const float *>(scratch.input.data);
    }

    scratch.class_ids.create(1, num_candidates, CV_32S);
    scratch.colors.create(1, num_candidates, CV_32S);
    scratch.confidences.create(1, num_candidates, CV_32F);
    scratch.keypoints.create(1, num_candidates * 8, CV_32F);
    scratch.boxes.create(1, num_candidates * 4, CV_32F);
    scratch.count.create(1, 1, CV_32S);
    scratch.count.setTo(cv::Scalar(0));

    decodeLidarKernel<<<gridSize(num_candidates), kBlockSize>>>(
        data_ptr, num_candidates, num_candidates,
        confidence_threshold,
        reinterpret_cast<int *>(scratch.count.data),
        reinterpret_cast<int *>(scratch.class_ids.data),
        reinterpret_cast<float *>(scratch.confidences.data),
        reinterpret_cast<float *>(scratch.boxes.data),
        reinterpret_cast<float *>(scratch.keypoints.data));
    cudaDeviceSynchronize();

    return collectArmorCandidates(scratch, infer_param, true);
}

CudaRuneCandidates cudaDecodeRune(const float *input_ptr, const InferParam &infer_param,
                                  float confidence_threshold,
                                  const void *device_input)
{
    const int num_candidates = infer_param.out_tensor_cols;
    const int num_features = infer_param.out_tensor_rows;

    ArmorScratch &scratch = armorScratch();
    const std::size_t total = static_cast<std::size_t>(num_candidates) * num_features;
    const float *data_ptr = nullptr;
    if (device_input != nullptr)
    {
        data_ptr = static_cast<const float *>(device_input);
    }
    else
    {
        scratch.input.create(1, static_cast<int>(total), CV_32F);
        cudaMemcpy(scratch.input.data, input_ptr, total * sizeof(float), cudaMemcpyHostToDevice);
        data_ptr = reinterpret_cast<const float *>(scratch.input.data);
    }

    scratch.class_ids.create(1, num_candidates, CV_32S);
    scratch.confidences.create(1, num_candidates, CV_32F);
    scratch.keypoints.create(1, num_candidates * 10, CV_32F);
    scratch.count.create(1, 1, CV_32S);
    scratch.count.setTo(cv::Scalar(0));

    decodeRuneKernel<<<gridSize(num_candidates), kBlockSize>>>(
        data_ptr, num_candidates, num_candidates,
        confidence_threshold,
        reinterpret_cast<int *>(scratch.count.data),
        reinterpret_cast<int *>(scratch.class_ids.data),
        reinterpret_cast<float *>(scratch.confidences.data),
        reinterpret_cast<float *>(scratch.keypoints.data));
    cudaDeviceSynchronize();

    CudaRuneCandidates result;

    cv::Mat host_count;
    scratch.count.download(host_count);
    const int candidate_count = host_count.at<int>(0);
    if (candidate_count <= 0)
        return result;

    cv::Mat host_class_ids;
    cv::Mat host_confidences;
    cv::Mat host_keypoints;
    scratch.class_ids.colRange(0, candidate_count).download(host_class_ids);
    scratch.confidences.colRange(0, candidate_count).download(host_confidences);
    scratch.keypoints.colRange(0, candidate_count * 10).download(host_keypoints);

    const int *class_ptr = host_class_ids.ptr<int>();
    const float *confidence_ptr = host_confidences.ptr<float>();
    const float *keypoint_ptr = host_keypoints.ptr<float>();

    result.class_ids.reserve(candidate_count);
    result.confidences.reserve(candidate_count);
    result.keypoints.reserve(static_cast<std::size_t>(candidate_count) * 5);

    for (int index = 0; index < candidate_count; ++index)
    {
        result.class_ids.push_back(class_ptr[index]);
        result.confidences.push_back(confidence_ptr[index]);
        for (int point = 0; point < 5; ++point)
        {
            const float x = keypoint_ptr[index * 10 + point * 2 + 0];
            const float y = keypoint_ptr[index * 10 + point * 2 + 1];
            result.keypoints.emplace_back((x - infer_param.pad_x) / infer_param.scale,
                                          (y - infer_param.pad_y) / infer_param.scale);
        }
    }

    return result;
}
} // namespace MPT
