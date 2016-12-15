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

#pragma once

#if __PARALLELTYPE__ == __GPU__

#include <cublas.h>
#include <cublas_api.h>

#include "cuda_log.h"

cudaError_t res;
cublasHandle_t handle;
cublasStatus_t status;

template<typename DTYPE>
inline DTYPE* cuda_malloc(int length)
{
	cublasInit();
	DTYPE* data_;
	res = cudaMalloc((void**) (&data_), length * sizeof(float_t));
	//status = cublasAlloc(length,sizeof(DTYPE),(void**)data_);
	//CUDA_CHECK(res);
	cublasShutdown();
	return data_;
}

template<typename DTYPE>
inline DTYPE* cuda_malloc_v(int length,DTYPE value)
{
	cublasInit();
	DTYPE* data_;
	vector<DTYPE> v(length,value);
	res = cudaMalloc((void**) (&data_), length * sizeof(float_t));
	//status = cublasAlloc(length,sizeof(DTYPE),(void**)data_);
	res = cudaMemcpy((void*) (data_), (void*) (&v[0]),	length * sizeof(DTYPE), cudaMemcpyHostToDevice);
	//CUDA_CHECK(res);
	cublasShutdown();
	return data_;
}

template<typename DTYPE>
inline void cuda_setvalue(DTYPE *data_,DTYPE value, int length)
{
	vector<DTYPE> v(length,value);
	res = cudaMemcpy((void*) (data_), (void*) (&v[0]),	length * sizeof(DTYPE), cudaMemcpyHostToDevice);
	//CUDA_CHECK(res);
}

template<typename DTYPE>
inline void cuda_copy2dev(DTYPE *data_,DTYPE* values, int length)
{
	res = cudaMemcpy((void*) (data_), (void*) (values),	length * sizeof(DTYPE), cudaMemcpyHostToDevice);
	//CUDA_CHECK(res);
}

template<typename DTYPE>
inline void cuda_free(DTYPE* data_)
{
	cudaFree(data_);
}

#endif
