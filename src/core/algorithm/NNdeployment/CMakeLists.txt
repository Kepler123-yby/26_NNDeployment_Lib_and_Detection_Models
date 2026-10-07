cmake_minimum_required(VERSION 3.22.0)
set(lib_name NNdeployment)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# 依赖路径配置（参考 qianli_vision 的直接查找方式）
#
# 每个依赖都“检测到即编译”：同一次构建会把可用的后端全部编进同一个库，
# 运行时由 YAML 配置的 deploy_way / preprocess_cuda / postprocess_cuda 选择，
# 因此切换后端或 CUDA 只需改配置文件，不需要重新编译。
#
#   -DNNDEPLOYMENT_OPENCV_CUDA_DIR=<带CUDA的OpenCV cmake目录>   # 首选（也认同名环境变量）
#   -DOpenCV_DIR=<OpenCV cmake 目录>                             # 通用 OpenCV 路径
#   -DOPENVINO_DIR=<OpenVINO runtime/cmake>                     # 也可用环境变量
#   -DTENSORRT_ROOT=<TensorRT 安装根目录>
# ============================================================================
set(OPENVINO_DIR "${OPENVINO_DIR}" CACHE PATH "OpenVINO runtime 的 cmake 目录（留空则用系统/环境变量）")
set(TENSORRT_ROOT "" CACHE PATH "TensorRT 安装根目录（含 include/ 与 lib/）")
set(NNDEPLOYMENT_OPENCV_CUDA_DIR "${NNDEPLOYMENT_OPENCV_CUDA_DIR}" CACHE PATH "带 CUDA 模块的 OpenCV cmake 目录（优先使用）")
option(NNDEPLOYMENT_ENABLE_OPENVINO "检测到 OpenVINO 时编译该后端" ON)
option(NNDEPLOYMENT_ENABLE_TENSORRT "检测到 TensorRT 时编译该后端" ON)
option(NNDEPLOYMENT_ENABLE_OPENCV_CUDA "检测到 OpenCV CUDA 模块时启用预处理/后处理加速" ON)

# ---------------------------------------------------------------------------
# 选择带 CUDA 的 OpenCV
# 常见坑：build/CMakeCache.txt 里残留 OpenCV_DIR 指向无 CUDA 的系统 OpenCV
# （如 /usr/lib/cmake/opencv4），会导致检测不到 CUDA。这里在 find_package 之前
# 探测候选安装，优先选中带 cudaarithm/cudawarping 的那一份，并纠正旧缓存。
# 判断方式是直接读 OpenCVModules.cmake，不做二次 find_package（否则会因导入
# 目标重复而报 “Some but not all targets ... already defined”）。
# ---------------------------------------------------------------------------
function(nn_opencv_dir_has_cuda dir out_var)
    set(${out_var} FALSE PARENT_SCOPE)
    if(NOT EXISTS "${dir}/OpenCVModules.cmake")
        return()
    endif()
    file(READ "${dir}/OpenCVModules.cmake" _nn_mods)
    if(_nn_mods MATCHES "opencv_cudawarping" AND _nn_mods MATCHES "opencv_cudaarithm")
        set(${out_var} TRUE PARENT_SCOPE)
    endif()
endfunction()

set(_nn_opencv_candidates)
if(DEFINED ENV{NNDEPLOYMENT_OPENCV_CUDA_DIR})
    list(APPEND _nn_opencv_candidates "$ENV{NNDEPLOYMENT_OPENCV_CUDA_DIR}")
endif()
if(NNDEPLOYMENT_OPENCV_CUDA_DIR)
    list(APPEND _nn_opencv_candidates "${NNDEPLOYMENT_OPENCV_CUDA_DIR}")
endif()
if(OpenCV_DIR)
    list(APPEND _nn_opencv_candidates "${OpenCV_DIR}")
endif()
list(APPEND _nn_opencv_candidates
    /usr/local/lib/cmake/opencv4
    /usr/local/lib/cmake/OpenCV
    /usr/lib/cmake/opencv4
    /usr/lib/aarch64-linux-gnu/cmake/opencv4
    /usr/lib/x86_64-linux-gnu/cmake/opencv4
)

if(NNDEPLOYMENT_ENABLE_OPENCV_CUDA)
    set(_nn_opencv_cuda_dir "")
    foreach(_nn_dir ${_nn_opencv_candidates})
        nn_opencv_dir_has_cuda("${_nn_dir}" _nn_has)
        if(_nn_has)
            set(_nn_opencv_cuda_dir "${_nn_dir}")
            break()
        endif()
    endforeach()

    if(_nn_opencv_cuda_dir)
        # 显式指定的 OpenCV_DIR（含旧缓存值）不含 CUDA 时，警告并改用带 CUDA 的那一份。
        if(OpenCV_DIR AND NOT OpenCV_DIR STREQUAL _nn_opencv_cuda_dir)
            nn_opencv_dir_has_cuda("${OpenCV_DIR}" _nn_cur_has)
            if(NOT _nn_cur_has)
                message(WARNING "NNdeployment: 指定的 OpenCV_DIR=${OpenCV_DIR} 不含 CUDA 模块，自动改用 ${_nn_opencv_cuda_dir}；如需强制使用指定版本，请加 -DNNDEPLOYMENT_ENABLE_OPENCV_CUDA=OFF")
            endif()
        endif()
        set(OpenCV_DIR "${_nn_opencv_cuda_dir}" CACHE PATH "OpenCV 配置目录（优先带 CUDA 模块）" FORCE)
        message(STATUS "NNdeployment: 选用带 CUDA 模块的 OpenCV：${_nn_opencv_cuda_dir}")
    else()
        message(STATUS "NNdeployment: 未找到带 CUDA 模块的 OpenCV（可用 -DNNDEPLOYMENT_OPENCV_CUDA_DIR=<...> 指定）")
    endif()
endif()

# OpenCV 为必需依赖。
find_package(OpenCV REQUIRED)
# 模型配置文件使用 YAML（与 qianli_vision 一致）。
find_package(yaml-cpp REQUIRED)

# CUDA Toolkit（TensorRT 与 OpenCV CUDA 都可能需要）。
find_package(CUDAToolkit QUIET)

# nvcc 不在 PATH 时（远程命令 / IDE / CI 的非登录 shell 常见），用 CUDAToolkit
# 找到的 nvcc 兜底设置 CMAKE_CUDA_COMPILER，否则 enable_language(CUDA) 会失败。
if(CUDAToolkit_FOUND AND CUDAToolkit_NVCC_EXECUTABLE AND EXISTS "${CUDAToolkit_NVCC_EXECUTABLE}")
    find_program(_nn_nvcc_in_path nvcc)
    if(NOT _nn_nvcc_in_path)
        message(STATUS "NNdeployment: nvcc 不在 PATH，改用 CUDAToolkit 提供的 ${CUDAToolkit_NVCC_EXECUTABLE}")
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA 编译器" FORCE)
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenVINO 后端
# ---------------------------------------------------------------------------
set(NNDEPLOYMENT_WITH_OPENVINO OFF)
if(NNDEPLOYMENT_ENABLE_OPENVINO)
    find_package(OpenVINO QUIET COMPONENTS Runtime)
    if(OpenVINO_FOUND)
        set(NNDEPLOYMENT_WITH_OPENVINO ON)
        message(STATUS "NNdeployment: 启用 OpenVINO 后端 (${OpenVINO_DIR})")
    else()
        message(STATUS "NNdeployment: 未找到 OpenVINO，跳过 OpenVINO 后端")
    endif()
endif()

# ---------------------------------------------------------------------------
# TensorRT 后端
# 优先使用 TensorRT 官方/第三方 CMake package；找不到时按 TENSORRT_ROOT 与
# 常见安装路径用 find_path/find_library 直接查找。
# ---------------------------------------------------------------------------
set(NNDEPLOYMENT_WITH_TENSORRT OFF)
if(NNDEPLOYMENT_ENABLE_TENSORRT AND CUDAToolkit_FOUND)
    find_package(TensorRT QUIET)
    if(NOT TensorRT_FOUND)
        find_path(TENSORRT_INCLUDE_DIR
            NAMES NvInfer.h
            PATHS ${TENSORRT_ROOT}/include
                  /usr/local/tensorrt/include
                  /usr/src/tensorrt/include
                  /usr/local/include
                  /usr/include/x86_64-linux-gnu
        )
        find_library(TENSORRT_NVINFER_LIB
            NAMES nvinfer
            PATHS ${TENSORRT_ROOT}/lib
                  ${TENSORRT_ROOT}/lib64
                  /usr/local/tensorrt/lib
                  /usr/src/tensorrt/lib
                  /usr/local/lib
                  /usr/lib/x86_64-linux-gnu
        )
        get_filename_component(_tensorrt_lib_dir "${TENSORRT_NVINFER_LIB}" DIRECTORY)
        find_library(TENSORRT_NVONNXPARSER_LIB
            NAMES nvonnxparser
            PATHS "${_tensorrt_lib_dir}"
                  ${TENSORRT_ROOT}/lib
                  ${TENSORRT_ROOT}/lib64
                  /usr/local/tensorrt/lib
                  /usr/src/tensorrt/lib
                  /usr/local/lib
                  /usr/lib/x86_64-linux-gnu
        )
        if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIB AND TENSORRT_NVONNXPARSER_LIB)
            set(TensorRT_FOUND TRUE)
            set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
            set(TENSORRT_LIBRARIES ${TENSORRT_NVINFER_LIB} ${TENSORRT_NVONNXPARSER_LIB})
        endif()
    endif()
    if(TensorRT_FOUND)
        set(NNDEPLOYMENT_WITH_TENSORRT ON)
        message(STATUS "NNdeployment: 启用 TensorRT 后端 (${TENSORRT_INCLUDE_DIR})")
    else()
        message(STATUS "NNdeployment: 未找到 TensorRT，跳过 TensorRT 后端")
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenCV CUDA 加速（预处理/后处理）
# 通过检查 OpenCV 是否提供 CUDA 模块目标来判断，无需单独传编译开关。
# ---------------------------------------------------------------------------
set(NNDEPLOYMENT_WITH_OPENCV_CUDA OFF)
if(NNDEPLOYMENT_ENABLE_OPENCV_CUDA AND CUDAToolkit_FOUND AND
   TARGET opencv_cudawarping AND TARGET opencv_cudaarithm)
    enable_language(CUDA)
    set(NNDEPLOYMENT_WITH_OPENCV_CUDA ON)
    message(STATUS "NNdeployment: 启用 OpenCV CUDA 预处理/后处理 (OpenCV ${OpenCV_VERSION} @ ${OpenCV_DIR})")
else()
    # 按原因分类提示，并给出修复命令，避免只打印一句容易被忽略的 STATUS。
    if(NOT NNDEPLOYMENT_ENABLE_OPENCV_CUDA)
        message(STATUS "NNdeployment: OpenCV CUDA 被手动关闭（-DNNDEPLOYMENT_ENABLE_OPENCV_CUDA=OFF）")
    elseif(NOT CUDAToolkit_FOUND)
        message(STATUS "NNdeployment: 未启用 OpenCV CUDA —— 未找到 CUDA Toolkit（安装 CUDA，或用 -DCUDAToolkit_ROOT=<...> 指定）")
    else()
        message(STATUS "NNdeployment: 未启用 OpenCV CUDA —— 当前 OpenCV 无 CUDA 模块（用 -DNNDEPLOYMENT_OPENCV_CUDA_DIR=<带 CUDA 的 OpenCV cmake 目录> 指定；若旧缓存把 OpenCV_DIR 固定在无 CUDA 版本，也可直接删掉 build/ 重配）")
    endif()
endif()

if(NOT NNDEPLOYMENT_WITH_OPENVINO AND NOT NNDEPLOYMENT_WITH_TENSORRT)
    message(FATAL_ERROR "未找到任何推理后端（OpenVINO 或 TensorRT），无法构建 NNdeployment_lib")
endif()

# 供 app_plugin 等子目录判断库中实际编译了哪些能力。
set(NNDEPLOYMENT_WITH_OPENVINO "${NNDEPLOYMENT_WITH_OPENVINO}" CACHE INTERNAL "NNdeployment: OpenVINO 后端已编译")
set(NNDEPLOYMENT_WITH_TENSORRT "${NNDEPLOYMENT_WITH_TENSORRT}" CACHE INTERNAL "NNdeployment: TensorRT 后端已编译")
set(NNDEPLOYMENT_WITH_OPENCV_CUDA "${NNDEPLOYMENT_WITH_OPENCV_CUDA}" CACHE INTERNAL "NNdeployment: OpenCV CUDA 已编译")

# ---------------------------------------------------------------------------
# 源文件：公共部分 + 检测到的后端
# ---------------------------------------------------------------------------
set(NNdeployment_src
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/network_deployment_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/network_deployment_lib.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/network_preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/postprocess/network_postprocess.cpp
)

if(NNDEPLOYMENT_WITH_OPENVINO)
    list(APPEND NNdeployment_src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/network_deployment_openvino.cpp
    )
endif()

if(NNDEPLOYMENT_WITH_TENSORRT)
    list(APPEND NNdeployment_src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/network_deployment_tensorrt.cpp
    )
endif()

if(NNDEPLOYMENT_WITH_OPENCV_CUDA)
    list(APPEND NNdeployment_src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/network_preprocess_cuda.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/postprocess/network_postprocess_cuda.cpp
    )
    # 这两个文件都包含 CUDA 设备代码（__global__ kernel 与 <<<>>> 启动），必须由 nvcc 编译。
    # 这里保持 .cpp 扩展名，但显式把它们的语言指定为 CUDA，仍交给 nvcc。
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/network_preprocess_cuda.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/postprocess/network_postprocess_cuda.cpp
        PROPERTIES LANGUAGE CUDA
    )
endif()

add_library(${lib_name}_lib SHARED ${NNdeployment_src})

if(NNDEPLOYMENT_WITH_OPENCV_CUDA)
    # 本目标同时含 C++ 与 CUDA 源码：交给 nvcc 链接，并把 CUDA 运行时设为共享库。
    # 否则 CMake 会用 CXX 链接器，并追加裸库名 cudadevrt;cudart，导致找不到 -lcudadevrt/-lcudart。
    set_target_properties(${lib_name}_lib PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
        LINKER_LANGUAGE CUDA
    )
endif()

add_library(${lib_name}_mpt_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/performance/network_performance.cpp
)
set_target_properties(${lib_name}_mpt_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${lib_name}_mpt_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/performance
)

# 头文件路径
target_include_directories(${lib_name}_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/interface
    ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(${lib_name}_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/preprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/postprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/performance
)

# 链接库
target_link_libraries(${lib_name}_lib PUBLIC
    ${OpenCV_LIBS}
)
target_link_libraries(${lib_name}_lib PRIVATE ${lib_name}_mpt_lib)
target_link_libraries(${lib_name}_lib PRIVATE yaml-cpp)

if(NNDEPLOYMENT_WITH_OPENVINO)
    target_link_libraries(${lib_name}_lib PRIVATE openvino::runtime)
endif()

if(NNDEPLOYMENT_WITH_TENSORRT)
    target_include_directories(${lib_name}_lib PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIRS}
    )
    target_link_libraries(${lib_name}_lib PRIVATE
        ${TENSORRT_LIBRARIES}
        CUDA::cudart
    )
endif()

if(NNDEPLOYMENT_WITH_OPENCV_CUDA)
    target_link_libraries(${lib_name}_lib PRIVATE CUDA::cudart)
endif()

# 把检测结果传给源码：1 表示该能力可用。
target_compile_definitions(${lib_name}_lib PRIVATE
    NNDEPLOYMENT_WITH_OPENVINO=$<BOOL:${NNDEPLOYMENT_WITH_OPENVINO}>
    NNDEPLOYMENT_WITH_TENSORRT=$<BOOL:${NNDEPLOYMENT_WITH_TENSORRT}>
    NNDEPLOYMENT_WITH_OPENCV_CUDA=$<BOOL:${NNDEPLOYMENT_WITH_OPENCV_CUDA}>
)

message(STATUS "NNdeployment 后端汇总: OpenVINO=${NNDEPLOYMENT_WITH_OPENVINO} TensorRT=${NNDEPLOYMENT_WITH_TENSORRT} OpenCV-CUDA=${NNDEPLOYMENT_WITH_OPENCV_CUDA}")
