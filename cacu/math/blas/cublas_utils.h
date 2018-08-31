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

#ifndef CUBLAS_UTILS_H_
#define CUBLAS_UTILS_H_

#include <stdio.h>
#include <stdlib.h>

#if defined(__USE_DEVICE__) && defined(__PARALLELTYPE__)
#if __USE_DEVICE__ == ON && __PARALLELTYPE__ == __CUDA__

#include "../../config.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cublas_v2.h"

namespace cacu {

extern cublasHandle_t handle;

//cublas log utilities
#define CUBLAS_LOG(level, status, log) \
				do{ \
					fprintf(stderr,"[%s][%s %s:%d] code %d :%s\n",level, __TIME__, __FILE__, __LINE__,status, log);  \
				}while(0); exit(-1);


#define CUBLAS_CHECK(status) \
		if(status!=CUBLAS_STATUS_SUCCESS) {\
			switch(status) { \
				case CUBLAS_STATUS_NOT_INITIALIZED: \
					CUBLAS_LOG("CUBLAS",status,"Resource allocation failed inside the cuBLAS library. This is usually caused by a cudaMalloc() failure."); \
					break; \
				case CUBLAS_STATUS_ALLOC_FAILED: \
					CUBLAS_LOG("CUBLAS",status,"An unsupported value or parameter was passed to the function (a negative vector size, for example)."); \
					break; \
				case CUBLAS_STATUS_ARCH_MISMATCH: \
					CUBLAS_LOG("CUBLAS",status,"The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision."); \
					break; \
				case CUBLAS_STATUS_MAPPING_ERROR: \
					CUBLAS_LOG("CUBLAS",status,"An access to GPU memory space failed, which is usually caused by a failure to bind a texture."); \
					break; \
				case CUBLAS_STATUS_EXECUTION_FAILED: \
					CUBLAS_LOG("CUBLAS",status,"The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons."); \
					break; \
				case CUBLAS_STATUS_INTERNAL_ERROR: \
					CUBLAS_LOG("CUBLAS",status,"An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure."); \
					break; \
				case CUBLAS_STATUS_NOT_SUPPORTED: \
					CUBLAS_LOG("CUBLAS",status,"The functionnality requested is not supported."); \
					break; \
				case CUBLAS_STATUS_LICENSE_ERROR: \
					CUBLAS_LOG("CUBLAS",status,"The functionnality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly."); \
					break; \
				default : \
					break; \
			} \
		}

void create_cublas_handle();

void release_cublas_handle();

}

#endif
#endif


#endif
