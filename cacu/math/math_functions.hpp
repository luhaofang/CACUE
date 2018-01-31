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

#include "math_definition.h"
#include "../config.h"
#include "../definition.h"

#include "cuda/math_functions_cuda.h"
#include "blas/math_function_oblas.hpp"
#include "blas/math_function_mkl.hpp"
#include "blas/math_function_cublas.hpp"
#include "utils/rand_t.h"

using namespace std;

namespace cacu {

/**
 * @cacu_saxpy
 * math y = a*x + y:
 * length: the input data's size
 */
inline void cacu_saxpy(float *x, const float a, float *y, const int length) {

#if __USE_DEVICE__ == OFF
#if __CBLASTYPE__ == __OPENBLAS__
	cacu_saxpy_oblas(x, a, y, length);
#elif __CBLASTYPE__ == __MKL__
	cacu_saxpy_mkl(x, a, y, length);
#endif
#else
#if __PARALLELTYPE__ == __CUDA__
	cacu_saxpy_cublas(x, a, y, length);
#endif
#endif
}

/**
 * @cacu_saxpby
 * math y = a*x + b*y:
 * length: the input data's size
 */
inline void cacu_saxpby(float *x, const float a, float *y, const float b,
		const int length) {
#if __USE_DEVICE__ == OFF
#if __CBLASTYPE__ == __OPENBLAS__
	cacu_saxpby_oblas(x, a, y, b, length);
#elif __CBLASTYPE__ == __MKL__
	cacu_saxpby_mkl(x, a, y, b, length);
#endif
#else
#if __PARALLELTYPE__ == __CUDA__
	cacu_saxpby_cublas(x, a, y, b, length);
#endif
#endif

}

/**
 * @cacu_scalex
 * math x[i] = a*x[i] :
 * x is a length dim array list, a is the corresponding scalar.
 */
inline void cacu_scalex(float *x, const int length, const float a) {

#if __USE_DEVICE__ == OFF
#if __CBLASTYPE__ == __OPENBLAS__
	cacu_scalex_oblas(x, a, length);
#elif __CBLASTYPE__ == __MKL__
	cacu_scalex_mkl(x, a, length);
#endif
#else
#if __PARALLELTYPE__ == __CUDA__
	cacu_scalex_cublas(x, a, length);
#endif
#endif

}

/**
 * @cacu_sgemv
 * math z = a*X*y + b*z:
 * trans_: whether x is needed to transpose
 */
inline void cacu_sgemv(TRANSPOSE trans_, float *x, const int x_height, float *y,
		const int x_width, const float alpha, float *z, const float beta) {

#if __USE_DEVICE__ == OFF
#if __CBLASTYPE__ == __OPENBLAS__
	CBLAS_TRANSPOSE transx =
	(trans_ == TRANS) ?
	CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	cacu_sgemv_oblas(transx, x, x_height, y, x_width, alpha, z, beta);
#elif __CBLASTYPE__ == __MKL__
	CBLAS_TRANSPOSE transx = (trans_ == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	cacu_sgemv_mkl(transx, x, x_height, y, x_width,alpha, z ,beta);
#endif
#else
#if __PARALLELTYPE__ == __CUDA__
	cublasOperation_t transx =
			(trans_ == TRANS) ?
					cublasOperation_t::CUBLAS_OP_T :
					cublasOperation_t::CUBLAS_OP_N;
	cacu_sgemv_cublas(transx, x, x_height, y, x_width, alpha, z, beta);
#endif
#endif
}

/**
 * @cacu_sgemm
 * math z = a*X*Y + b*z:
 * transx_: whether x is need to transpose
 * transy_: whether y is need to transpose((rand() % ( max - min + 1)) + min);
 */
inline void cacu_sgemm(TRANSPOSE transx_, TRANSPOSE transy_, float *x,
		const int x_height, const int x_width, float *y, const int y_width,
		const float alpha, float *z, const float beta) {
#if __USE_DEVICE__ == OFF
#if __CBLASTYPE__ == __OPENBLAS__
	CBLAS_TRANSPOSE transx =
	(transx_ == TRANS) ?
	CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	CBLAS_TRANSPOSE transy =
	(transy_ == TRANS) ?
	CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	cacu_sgemm_oblas(transx, transy, x, x_height, x_width, y, y_width, alpha, z,
			beta);
#elif __CBLASTYPE__ == __MKL__
	CBLAS_TRANSPOSE transx = (transx_ == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	CBLAS_TRANSPOSE transy = (transy_ == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	cacu_sgemm_mkl(transx, transy, x, x_height, x_width, y, y_width, alpha, z, beta);
#endif
#else
#if __PARALLELTYPE__ == __CUDA__
	cublasOperation_t transx =
			(transx_ == TRANS) ?
					cublasOperation_t::CUBLAS_OP_T :
					cublasOperation_t::CUBLAS_OP_N;
	cublasOperation_t transy =
			(transy_ == TRANS) ?
					cublasOperation_t::CUBLAS_OP_T :
					cublasOperation_t::CUBLAS_OP_N;
	cacu_sgemm_cublas(transx, transy, x, x_height, x_width, y, y_width, alpha, z,
			beta);
#endif
#endif
}

/**
 * @cacu_saxpy_atomic
 * math y = ax + y:
 * length: the input data's size
 */
inline void cacu_saxpy_atomic(float *x, const float a, float *y,
		const int length) {
#if __USE_DEVICE__ == OFF
#if __CBLASTYPE__ == __OPENBLAS__
	cacu_saxpy_oblas(x, a, y, length);
#elif __CBLASTYPE__ == __MKL__
	cacu_saxpy_mkl(x, a, y, length);
#endif
#else
#if __PARALLELTYPE__ == __CUDA__
	cacu_saxpy_atomic_cuda(x, a, y, length);
#endif
#endif
}

/**
 * @rand_vector
 * math vector_[i] = (rand()>=ratio_)
 * length: the input data's size
 */
void rand_vector(float *vector_, const int length, const float ratio_);

/**
 * @mask_vector
 * math vector_[i] = (vector_[i]>0)
 * length: the input data's size
 */
void mask_vector(float *vector_, const int length, float *mask);

/**
 * @cacu_copy
 * math y = x:
 * length: the input data's size
 */
void cacu_copy(float *x, const int length, float *y);

/**
 * @cacu_isaxdb
 * y[index] = x[index]*a + b
 */
void cacu_isaxb(float *x, const int length, const float a,
		unsigned int *index_, const float b, float *y);

unsigned int argmax(float *data, const int length);

void cacu_transpose(float *mtx, const int m, const int n);

void cacu_clip_vec(float *data, const float threshold, const int length);

}
