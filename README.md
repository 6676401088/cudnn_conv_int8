# How to run
Compile and run `src/cudnn_conv_float32.cc` or `src/cudnn_conv_int8.cc` with CUDA 8.0 and cuDNN 6.0.
The code is self contained and all the parameters are hardcoded into the code to help debugging the propblem.

## Working version (32 bit float)
`src/cudnn_conv_float32.cc` is a simple implementation of FLOAT32 convolution using cuDNN 6. This code seems to work.

## Broken version (8 bit int)
`src/cudnn_conv_int8.cc` is a variant of the above FLOAT32 version for INT8-based convolution. As explained in the user manual, you must have compute capability of 6.1 or higher. A number of parameters are changed from the FLOAT32 version following the [user manual](http://developer2.download.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/Doc/CUDNN_Library.pdf?afFAtswqyLpPoty-E55PJd8z1XC5RyERCZXGEJ5jvCTE7vMVYuLFkakbXKfWx-NdE27mCVVQ2i6MGpgT5wc5u3XW1AVg00dHW8z7ZUmoG1-7eY4TilRkaFceS00cunWAPShEEa1SxeSRHIdImuUF8d232eOPsDqVtPLbftEYFF6Nag). **This code fails with `CUDNN_STATUS_NOT_SUPPORTED` error**.
* Descriptor data types are set to `CUDNN_DATA_INT8` for the input/output tensors and filter (See page 59)
* Convolution descriptor has data type `CUDNN_DATA_INT32`, as instructed in the manual (See page 59)
* Data format for the input/output tensor descriptors and the filter are changed to `CUDNN_TENSOR_NHWC` (See page 62)

We can reproduce the problem on GTX 1070, Titan X (Pascal), and Quadro 6000. 

# Additional information
* `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMPUTED_GEMM` (see page 62) is **not** supported in `cudnnConvolutionFwdAlgo_t`. The closest alternative seems to be `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` ("PRECOMP" instead of "PRECOMPUTED")
* The job fails at cudnnConvolutionForward() with `CUDNN_STATUS_NOT_SUPPORTED` error. This happens regardless of what algorithm I choose. I tested all algo types in page 16 of the manual.
* FLOAT32 implementation (`CUDNN_DATA_FLOAT`) doesn't have this issue

Please see my [post](https://devtalk.nvidia.com/default/topic/1005119/cudnn-v6-int8-convolution-failing-with-cudnn_status_not_supported/) on the developer forum.
