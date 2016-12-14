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

#include "math_function_openmp.h"
#include "math_function_oblas.h"

#include "cuda/math_functions_cuda.h"

namespace mycnn{


inline void cacu_saxpy(float_t *x, float_t a, float_t *y,int length)
{
#if __PARALLELTYPE__ == __OPENMP__
	cacu_saxpby_omp(x, a, y, length);
#elif __PARALLELTYPE__ == __OPENBLAS__
	cacu_saxpy_oblas(x, a, y, length);
#elif __PARALLELTYPE__ == __GPU__
	cacu_saxpy_gpu(x, a, y, length);
#endif
}

inline void cacu_saxpby(float_t *x, float_t a, float_t *y, float_t b, int length)
{
#if __PARALLELTYPE__ == __OPENMP__
	cacu_saxpby_omp(x, a, y, b, length);
#elif __PARALLELTYPE__ == __OPENBLAS__
	cacu_saxpby_oblas(x, a, y, b, length);
#elif __PARALLELTYPE__ == __GPU__
	cacu_saxpby_gpu(x, a, y, b, length);
#endif

}


inline void cacu_sgemv(TRANSPOSE trans,float_t *x, int x_height, float_t *y, int x_width, float_t *z)
{
#if __PARALLELTYPE__ == __OPENMP__
	cacu_sgemv_omp(x, x_height, y, x_width, z);
#elif __PARALLELTYPE__ == __OPENBLAS__
	CBLAS_TRANSPOSE transx = (trans == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	cacu_sgemv_oblas(transx, x, x_height, y, x_width, z);
#elif __PARALLELTYPE__ == __GPU__
	cublasOperation_t transx = (trans == TRANS) ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N;
	cacu_sgemv_gpu(transx, x, x_height, y, x_width, z);
#endif
}


inline void cacu_sgemm(TRANSPOSE transx_, TRANSPOSE transy_, float_t *x, int x_height, int x_width, float_t *y, int y_height, float_t *z)
{
#if __PARALLELTYPE__ == __OPENMP__
	cacu_sgemm_omp(transx, transy, x, x_height, x_width, y, y_height, z);
#elif __PARALLELTYPE__ == __OPENBLAS__
	CBLAS_TRANSPOSE transx = (transx_ == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	CBLAS_TRANSPOSE transy = (transy_ == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	cacu_sgemm_oblas(transx, transy, x, x_height, x_width, y, y_height, z);
#elif __PARALLELTYPE__ == __GPU__
	cublasOperation_t transx = (transx_ == TRANS) ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N;
	cublasOperation_t transy = (transy_ == TRANS) ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N;
	cacu_sgemm_gpu(transx, transy, x, x_height, x_width, y, y_height, z);
#endif
}


inline void cacu_copy(float_t *x, int length, float_t *y)
{
#if __PARALLELTYPE__ == __GPU__
	cacu_copy_gpu(x, length, y);
#else
	for(int i = 0 ; i < length ; ++i)
		y[i] = x[i];
#endif
}


};
