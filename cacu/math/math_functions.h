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

#ifndef MATH_FUCNTIONS_H_
#define MATH_FUCNTIONS_H_

#include "cpu/math_functions_cpu.h"
#include "cuda/math_functions_cuda.h"
#include "blas/math_function_oblas.h"
#include "blas/math_function_mkl.h"
#include "blas/math_function_cublas.h"
#include "blas/math_function_naive.h"
#include "utils/rand_t.h"

using namespace std;

namespace cacu {

/**
 * @cacu_saxpy
 * math y = a*x + y:
 * length: the input data's size
 */
inline void cacu_saxpy(float_t *x, const float_t a, float_t *y, const int length)
{
#if __USE_DEVICE__ == OFF
#if __CBLASTYPE__ == __OPENBLAS__
		cacu_saxpy_oblas(x, a, y, length);
#elif __CBLASTYPE__ == __MKL__
		cacu_saxpy_mkl(x, a, y, length);
#elif __CBLASTYPE__ == __NAIVE__
		cacu_saxpy_naive(x, a, y, length);
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
inline void cacu_saxpby(float_t *x, const float_t a, float_t *y, const float_t b,
	const int length){
#if __USE_DEVICE__ == OFF
#if __CBLASTYPE__ == __OPENBLAS__
		cacu_saxpby_oblas(x, a, y, b, length);
#elif __CBLASTYPE__ == __MKL__
		cacu_saxpby_mkl(x, a, y, b, length);
#elif __CBLASTYPE__ == __NAIVE__
		cacu_saxpby_naive(x, a, y, b, length);
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
inline void cacu_scalex(float_t *x, const int length, const float_t a){
#if __USE_DEVICE__ == OFF
#if __CBLASTYPE__ == __OPENBLAS__
		cacu_scalex_oblas(x, a, length);
#elif __CBLASTYPE__ == __MKL__
		cacu_scalex_mkl(x, a, length);
#elif __CBLASTYPE__ == __NAIVE__
		cacu_scalex_naive(x, a, length);
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
inline void cacu_sgemv(TRANSPOSE trans_, float_t *x, const int x_height, float_t *y,
	const int x_width, const float_t alpha, float_t *z, const float_t beta){
#if __USE_DEVICE__ == OFF
#if __CBLASTYPE__ == __OPENBLAS__
		CBLAS_TRANSPOSE transx =
			(trans_ == TRANS) ?
			CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
		cacu_sgemv_oblas(transx, x, x_height, y, x_width, alpha, z, beta);
#elif __CBLASTYPE__ == __MKL__
		CBLAS_TRANSPOSE transx =
			(trans_ == TRANS) ?
			CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
		cacu_sgemv_mkl(transx, x, x_height, y, x_width, alpha, z, beta);
#elif __CBLASTYPE__ == __NAIVE__
		cacu_sgemv_naive(trans_, x, x_height, y, x_width, alpha, z, beta);
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
inline void cacu_sgemm(TRANSPOSE transx_, TRANSPOSE transy_, float_t *x,
	const int x_height, const int x_width, float_t *y, const int y_width,
	const float_t alpha, float_t *z, const float_t beta){
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
		CBLAS_TRANSPOSE transx =
			(transx_ == TRANS) ?
			CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
		CBLAS_TRANSPOSE transy =
			(transy_ == TRANS) ?
			CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
		cacu_sgemm_mkl(transx, transy, x, x_height, x_width, y, y_width, alpha, z,
			beta);
#elif __CBLASTYPE__ == __NAIVE__
		cacu_sgemm_naive(transx_, transy_, x, x_height, x_width, y, y_width, alpha, z,
			beta);
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
inline void cacu_saxpy_atomic(float_t *x, const float_t a, float_t *y,
	const int length){
#if __USE_DEVICE__ == OFF
#if __CBLASTYPE__ == __OPENBLAS__
		cacu_saxpy_oblas(x, a, y, length);
#elif __CBLASTYPE__ == __MKL__
		cacu_saxpy_mkl(x, a, y, length);
#elif __CBLASTYPE__ == __NAIVE__
		cacu_saxpy_naive(x, a, y, length);
#endif
#else
#if __PARALLELTYPE__ == __CUDA__
		cacu_saxpy_atomic_cuda(x, a, y, length);
#endif
#endif
}

/**
 * @cacu_saxpy_cpu
 * math y = ax + y:
 * length: the input data's size
 */
inline void cacu_saxpy_cpu(float_t *x, const float_t a, float_t *y,
	const int length){
#if __CBLASTYPE__ == __OPENBLAS__
		cacu_saxpy_oblas(x, a, y, length);
#elif __CBLASTYPE__ == __MKL__
		cacu_saxpy_mkl(x, a, y, length);
#elif __CBLASTYPE__ == __NAIVE__
		cacu_saxpy_naive(x, a, y, length);
#endif
}

/**
 * @rand_vector
 * math vector_[i] = (rand()>=ratio_)
 * length: the input data's size
 */
inline void rand_vector(float_t *vector_, const int length, const float_t ratio_) {

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	vec_t v_(length);

	for (int i = 0; i < length; ++i) {
		if (urand(0, 1) >= ratio_)
			v_[i] = 1;
	}
	cuda_copy2dev(vector_, &v_[0], length);
	vec_t().swap(v_);
//	rand_vector_cuda(vector_, length, ratio_);
#endif
#else
	rand_vector_cpu(vector_, length, ratio_);
#endif

}

/**
 * @mask_vector
 * math vector_[i] = (vector_[i]>0)
 * length: the input data's size
 */
inline void mask_vector(float_t *vector_, const int length, float_t *mask) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	vec_t v_(length);
	cuda_copy2host(&v_[0], vector_, length);
	for (int i = 0; i < length; ++i) {
		v_[i] = 1.0 * (v_[i] > 0);
	}
	cuda_copy2dev(mask, &v_[0], length);
	vec_t().swap(v_);
#endif
#else
	mask_vector_cpu(vector_, length, mask);
#endif
}

/**
* @mask_vector
* math vector_[i] = (vector_[i]<0)
* length: the input data's size
*/
inline void mask_vector_lt(float_t *vector_, const int length, float_t *mask) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	vec_t v_(length);
	cuda_copy2host(&v_[0], vector_, length);
	for (int i = 0; i < length; ++i) {
		v_[i] = 1.0 * (v_[i] < 0);
	}
	cuda_copy2dev(mask, &v_[0], length);
	vec_t().swap(v_);
#endif
#else
	mask_vector_lt_cpu(vector_, length, mask);
#endif
}

/**
 * @cacu_isaxdb
 * y[index] = x[index]*a + b
 */
inline void cacu_isaxb(float_t *x, const int channel, const int width, const int height, const float_t a, int *index_,
		const float_t b, float_t *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_isaxb_cuda(x, channel, width, height, a, index_, b, y);
#endif
#else
	cacu_isaxb_cpu(x, channel, width, height, a, index_, b, y);
#endif
}

inline unsigned int argmax(float_t *data, const int length) {

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	unsigned int index;
	vector<float_t> _data(length);
	cuda_copy2host(&_data[0], data, length);
	float_t max;
	unsigned int i;
	max = _data[0];
	index = 0;
	for (i = 1; i < length; ++i) {
		if (_data[i] > max) {
			max = _data[i];
			index = i;
		}
	}
	vector<float_t>().swap(_data);
	return index;
#endif
#else
	return argmax_cpu(data, length);
#endif
}

inline void cacu_transpose(float_t *mtx, const int m, const int n, const int clength) {

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_transpose_cuda(mtx, m, n, clength);
#endif
#else
	cacu_transpose_cpu(mtx, m, n, clength);
#endif
}

inline void cacu_rotate(float_t *mtx, const int m, const int n, const int clength, rotate_code rotate){
	assert(m == n);
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_rotate_cuda(mtx, m, n, clength, rotate);
#endif
#else
	cacu_rotate_cpu(mtx, m, n, clength, rotate);
#endif
}


/**
 * @cacu_copy
 * math y = x:
 * length: the input data's size
 */
inline void cacu_copy(float_t *x, const int length, float_t *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_copy_cublas(x, length, y);
#endif
#else
	cacu_copy_cpu(x, length, y);
#endif
}

/**
 * @cacu_clip_vec
 * math vector_[i] belongs to [lt, rt]
 * length: the input data's size
 */
inline void cacu_clip_vec(float_t *data, const float_t l_t, const float_t r_t, const int length) {
	assert(l_t < r_t);
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_clip_vec_cuda(data, l_t, r_t, length);
#endif
#else
	cacu_clip_vec_cpu(data, l_t, r_t, length);
#endif
}


inline void cacu_abs(float_t *x, const int length, float_t *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_abs_cuda(x, length, y);
#endif
#else
	cacu_abs_cpu(x, length, y);
#endif
}

inline void cacu_abs_grad(float_t *x, float_t *diff, const int length) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_abs_grad_cuda(x, diff, length);
#endif
#else
	cacu_abs_grad_cpu(x, diff, length);
#endif
}

inline void cacu_pow(float_t *x, const int length, const float_t pow, float_t *y){
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_pow_cuda(x, length, pow, y);
#endif
#else
	cacu_pow_cpu(x, length, pow, y);
#endif
}

inline void cacu_powt(float_t *x, const int length, const float_t *pow, float_t *y){
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_powt_cuda(x, length, pow, y);
#endif
#else
	cacu_powt_cpu(x, length, pow, y);
#endif
}

inline void cacu_ln(float_t *x, const int length, float_t *y){
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_ln_cuda(x, length, y);
#endif
#else
	cacu_ln_cpu(x, length, y);
#endif
}

inline void cacu_flip(int *x, const int length){
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_flip_cuda(x, length);
#endif
#else
	cacu_flip_cpu(x, length);
#endif
}

inline void cacu_cos(float_t *x, const int length, float_t *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_cos_cuda(x, length, y);
#endif
#else
	cacu_cos_cpu(x, length, y);
#endif
}

inline void cacu_acos(float_t *x, const int length, float_t *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_acos_cuda(x, length, y);
#endif
#else
	cacu_acos_cpu(x, length, y);
#endif
}

inline void cacu_floor(float_t *x, const int length, float_t *	y){
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_floor_cuda(x, length, y);
#endif
#else
	cacu_floor_cpu(x, length, y);
#endif
}


inline float_t cacu_normal_cdf(float_t value, float_t m = 0, float_t std = 1){
	value = (value - m) / std;
	return 0.5 * erfc(-value * sqrtf(0.5));
}

}


#endif
