/*
 Copyright (c) 2016, David lu
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of the <organization> nor the
 names of its contributors may be used to endorse or promote products
 derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CUDNN_LOG_H_
#define CUDNN_LOG_H_

#include <stdio.h>
#include <stdlib.h>

#include "../../config.h"

#if __USE_DEVICE__ == ON
#if __USE_CUDNN__ == ON

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cudnn.h>

namespace cacu {

extern cudnnHandle_t CUDNN_HANDLE;

//cudnn log utilities
#define CUDNN_LOG(level, status, log) \
				do{ \
					fprintf(stderr,"[%s][%s %s:%d] code %d :%s\n",level, __TIME__, __FILE__, __LINE__,status, log);  \
				}while(0); exit(-1);


#define CUDNN_CHECK(status) \
		if(status!=CUDNN_STATUS_SUCCESS) {\
			switch(status) { \
				case CUDNN_STATUS_NOT_INITIALIZED: \
					CUDNN_LOG("CUDNN",status,"The cuDNN library was not initialized properly. This error is usually returned when a call to cudnnCreate() fails or when cudnnCreate() has not been called prior to calling another cuDNN routine. In the former case, it is usually due to an error in the CUDA Runtime API called by cudnnCreate() or by an error in the hardware setup."); \
					break; \
				case CUDNN_STATUS_ALLOC_FAILED: \
					CUDNN_LOG("CUDNN",status,"Resource allocation failed inside the cuDNN library. This is usually caused by an internal cudaMalloc() failure."); \
					break; \
				case CUDNN_STATUS_BAD_PARAM: \
					CUDNN_LOG("CUDNN",status,"An incorrect value or parameter was passed to the function."); \
					break; \
				case CUDNN_STATUS_ARCH_MISMATCH: \
					CUDNN_LOG("CUDNN",status,"The function requires a feature absent from the current GPU device. Note that cuDNN only supports devices with compute capabilities greater than or equal to 3.0."); \
					break; \
				case CUDNN_STATUS_MAPPING_ERROR: \
					CUDNN_LOG("CUDNN",status,"An access to GPU memory space failed, which is usually caused by a failure to bind a texture."); \
					break; \
				case CUDNN_STATUS_EXECUTION_FAILED: \
					CUDNN_LOG("CUDNN",status,"The GPU program failed to execute. This is usually caused by a failure to launch some cuDNN kernel on the GPU, which can occur for multiple reasons."); \
					break; \
				case CUDNN_STATUS_INTERNAL_ERROR: \
					CUDNN_LOG("CUDNN",status,"An internal cuDNN operation failed."); \
					break; \
				case CUDNN_STATUS_NOT_SUPPORTED: \
					CUDNN_LOG("CUDNN",status,"The functionality requested is not presently supported by cuDNN."); \
					break; \
				case CUDNN_STATUS_LICENSE_ERROR: \
					CUDNN_LOG("CUDNN",status,"The functionality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly."); \
					break; \
				case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING: \
					CUDNN_LOG("CUDNN",status,"Runtime library required by RNN calls (libcuda.so or nvcuda.dll) cannot be found in predefined search paths."); \
					break; \
				case CUDNN_STATUS_RUNTIME_IN_PROGRESS: \
					CUDNN_LOG("CUDNN",status,"Some tasks in the user stream are not completed."); \
					break; \
				case CUDNN_STATUS_RUNTIME_FP_OVERFLOW: \
					CUDNN_LOG("CUDNN",status,"Numerical overflow occurred during the GPU kernel execution."); \
					break; \
				default : \
					break; \
			} \
		}

extern void create_cudnn_handle();

extern void release_cudnn_handle();

}

#endif
#endif


#endif
