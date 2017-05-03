#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include <ctime>
#include <cfloat>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>

#include "cuda.h"
#include "cudnn.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/** Error handling from https://developer.nvidia.com/cuDNN */
#define FatalError(s)                                                          \
  do {                                                                         \
    std::stringstream _where, _message;                                        \
    _where << __FILE__ << ':' << __LINE__;                                     \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;          \
    std::cerr << _message.str() << "\nAborting...\n";                          \
    cudaDeviceReset();                                                         \
    exit(1);                                                                   \
  } while (0)

#define checkCUDNN(status)                                                     \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);              \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

#define checkCudaErrors(status)                                                \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != 0) {                                                         \
      _error << "Cuda failure: " << status;                                    \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

/** Convolutional layer */
struct ConvolutionLayer {
  int kernel_size;
  int in_channels, in_height, in_width;
  int out_channels, out_height, out_width;
  std::vector<float> pconv;

  ConvolutionLayer(int in_channels_,
                   int out_channels_,
                   int kernel_size_,
                   int in_w_,
                   int in_h_)
    : pconv(in_channels_ * kernel_size_ * kernel_size_ * out_channels_) {
    in_channels = in_channels_;
    out_channels = out_channels_;
    kernel_size = kernel_size_;
    in_width = in_w_;
    in_height = in_h_;
    out_width = in_w_ - kernel_size_ + 1;
    out_height = in_h_ - kernel_size_ + 1;
  }
};

/** Training context */
struct TrainingContext {
  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor;
  cudnnFilterDescriptor_t conv1filterDesc;
  cudnnConvolutionDescriptor_t conv1Desc;
  cudnnConvolutionFwdAlgo_t conv1algo;
  int m_gpuid;
  int m_batchSize;
  size_t m_workspaceSize;

  // Disable copying
  TrainingContext& operator=(const TrainingContext&) = delete;
  TrainingContext(const TrainingContext&) = delete;

  // Constructor
  TrainingContext(int gpuid, int batch_size, ConvolutionLayer& conv1)
    : m_gpuid(gpuid) {
    m_batchSize = batch_size;

    /** Create descriptors within the constructor.
      * As instructed in the Usual manual, descriptors for
      * input and output tensors, filter, and the forward
      * convolution operator are created along with
      * cuDNN handle.
      */
    checkCudaErrors(cudaSetDevice(gpuid));
    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
    checkCUDNN(cudnnCreateFilterDescriptor(&conv1filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));

    // Initialize convolution forward pass
    size_t workspaceSizeFromConv = SetFwdConvolutionTensors(
        conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo);
    m_workspaceSize = std::max((int)workspaceSizeFromConv, 0);
  }

  ~TrainingContext() {
    checkCudaErrors(cudaSetDevice(m_gpuid));
    checkCUDNN(cudnnDestroy(cudnnHandle));

    checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(conv1Tensor));
    checkCUDNN(cudnnDestroyFilterDescriptor(conv1filterDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1Desc));
  }

  /** Set tensors and ops for forward pass */
  size_t SetFwdConvolutionTensors(ConvolutionLayer& conv,
                                  cudnnTensorDescriptor_t& srcTensorDesc,
                                  cudnnTensorDescriptor_t& dstTensorDesc,
                                  cudnnFilterDescriptor_t& filterDesc,
                                  cudnnConvolutionDescriptor_t& convDesc,
                                  cudnnConvolutionFwdAlgo_t& algo) {
    int n = m_batchSize;
    int c = conv.in_channels;
    int h = conv.in_height;
    int w = conv.in_width;

    // Set input tensor. Folowing the manual, chagnged
    // * CUDNN_DATA_FLOAT -> CUDNN_DATA_INT8, and 
    // * CUDNN_TENSOR_NCHW -> CUDNN_TENSOR_NHWC
    checkCUDNN(cudnnSetTensor4dDescriptor(
        srcTensorDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8, n, c, h, w));

    // Set convolution filter. Folowing the manual, chagnged
    // * CUDNN_DATA_FLOAT -> CUDNN_DATA_INT8, and 
    // * CUDNN_TENSOR_NCHW -> CUDNN_TENSOR_NHWC
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                          CUDNN_DATA_INT8,
                                          CUDNN_TENSOR_NHWC,
                                          conv.out_channels,
                                          conv.in_channels,
                                          conv.kernel_size,
                                          conv.kernel_size));

    // Set convolution operator. Folowing the manual, chagnged
    // * CUDNN_DATA_FLOAT -> CUDNN_DATA_INT32
    int pad_height = 0;
    int pad_width = 0;
    int stride_h = 1;
    int stride_v = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                               pad_height,
                                               pad_width,
                                               stride_h,
                                               stride_v,
                                               dilation_h,
                                               dilation_w,
                                               CUDNN_CONVOLUTION,
                                               CUDNN_DATA_INT32));

    // Compute output dimension. Folowing the manual, chagnged
    // * CUDNN_DATA_FLOAT -> CUDNN_DATA_INT8, and 
    // * CUDNN_TENSOR_NCHW -> CUDNN_TENSOR_NHWC
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        convDesc, srcTensorDesc, filterDesc, &n, &c, &h, &w));

    // Set output tensor (Changed CUDNN_DATA_FLOAT to CUDNN_DATA_INT8, following the manual)
    checkCUDNN(cudnnSetTensor4dDescriptor(
        dstTensorDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8, n, c, h, w));

    // Retrieve orward pass algorithm. We can either hardcode it to a specific
    // algorithm or use cudnnGetConvolutionForwardAlgorithm. For the purpose
    // of this test, either way works.
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    // Following also works
    // checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
    //     cudnnHandle,
    //     srcTensorDesc,
    //     filterDesc,
    //     convDesc,
    //     dstTensorDesc,
    //     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    //     0,
    //     &algo));
    

    // Compute workspace size. We can either hardcode it to a specific number,
    // or use cudnnGetConvolutionForwardWorkspaceSize. For the purpose of this
    // test, either way works.
    size_t sizeInBytes = 1073741824;    
    // Following also works
    // size_t sizeInBytes = 0;
    // checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
    //                                                    srcTensorDesc,
    //                                                    filterDesc,
    //                                                    convDesc,
    //                                                    dstTensorDesc,
    //                                                    algo,
    //                                                    &sizeInBytes));
    

    return sizeInBytes;
  }

  /** Execute forward pass */
  void ForwardPropagation(float* data,
                          float* conv1,
                          float* pconv1,
                          void* workspace) {
    float alpha = 1.0f;
    float beta = 0.0f;
    checkCudaErrors(cudaSetDevice(m_gpuid));
    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                                       &alpha,
                                       dataTensor,
                                       data,
                                       conv1filterDesc,
                                       pconv1,
                                       conv1Desc,
                                       conv1algo,
                                       workspace,
                                       m_workspaceSize,
                                       &beta,
                                       conv1Tensor,
                                       conv1));
  }
};

int main() {
  // parameters
  int gpu = 0;
  int iterations = 10000;

  // input dimensions
  size_t width = 960;
  size_t height = 600;
  size_t channels = 3;
  int batch_size = 1;

  // Create layer architecture
  int out_channels = 1;
  int kernel_size = 3;
  ConvolutionLayer conv1(
      (int)channels, out_channels, kernel_size, (int)width, (int)height);
  TrainingContext context(gpu, batch_size, conv1);

  // Initizlie convolution weight
  std::mt19937 g(42);
  float wconv1 =
      sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
  std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
  for (auto&& iter : conv1.pconv) {
    iter = static_cast<float>(dconv1(g));
  }

  // Initailize input image (batch size = 1)
  std::vector<float> img_float(1 * width * height * channels);
  for (auto&& iter : img_float) {
    iter = static_cast<float>(dconv1(g));
  }

  // Allocate input and output on GPU; copy input over to GPU
  float* d_data, *d_conv1;
  checkCudaErrors(cudaMalloc(&d_data,
                             sizeof(float) * context.m_batchSize * channels *
                                 height * width));
  checkCudaErrors(cudaMalloc(&d_conv1,
                             sizeof(float) * context.m_batchSize *
                                 conv1.out_channels * conv1.out_height *
                                 conv1.out_width));
  checkCudaErrors(cudaMemcpyAsync(d_data,
                                  &img_float[0],
                                  sizeof(float) * 1 * channels * width * height,
                                  cudaMemcpyHostToDevice));

  // Allocate kernel on GPU
  float* d_pconv1;
  checkCudaErrors(cudaMalloc(&d_pconv1, sizeof(float) * conv1.pconv.size()));
  checkCudaErrors(cudaMemcpyAsync(d_pconv1,
                                  &conv1.pconv[0],
                                  sizeof(float) * conv1.pconv.size(),
                                  cudaMemcpyHostToDevice));

  // Temporary buffers and workspaces
  void* d_cudnn_workspace = nullptr;
  if (context.m_workspaceSize > 0) {
    checkCudaErrors(cudaMalloc(&d_cudnn_workspace, context.m_workspaceSize));
  }

  // Start forward pass
  printf("Begin forwrad pass\n");
  checkCudaErrors(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iterations; ++iter) {
    context.ForwardPropagation(d_data, d_conv1, d_pconv1, d_cudnn_workspace);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  auto t2 = std::chrono::high_resolution_clock::now();

  printf(
      "Iteration time: %f ms\n",
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
          1000.0f / iterations);

  // Free data structures
  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFree(d_conv1));
  checkCudaErrors(cudaFree(d_pconv1));

  if (d_cudnn_workspace != nullptr)
    checkCudaErrors(cudaFree(d_cudnn_workspace));

  return 0;
}