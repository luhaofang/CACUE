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

#include "cuda/cuda_utils.h"

#include "math_function_openmp.h"
#include "math_function_oblas.h"
#include "math_function_mkl.h"

#include "cuda/math_functions_cuda.h"

#include "../utils/configs.h"
#include "../utils/data_defination.h"

namespace mycnn{

/**
 * @cacu_saxpy
 * math y = a*x + y:
 * length: the input data's size
 */
inline void cacu_saxpy(const float *x,const float a, float *y,int length)
{

#if __PARALLELTYPE__ == __OPENBLAS__
	cacu_saxpy_oblas(x, a, y, length);
#elif __PARALLELTYPE__ == __MKL__
	cacu_saxpy_mkl(x, a, y, length);
#elif __PARALLELTYPE__ == __CUDA__
	cacu_saxpy_gpu(x, a, y, length);
#endif
}

/**
 * @cacu_saxpby
 * math y = a*x + b*y:
 * length: the input data's size
 */
inline void cacu_saxpby(const float *x,const float a, float *y,const float b, int length)
{

#if __PARALLELTYPE__ == __OPENBLAS__
	cacu_saxpby_oblas(x, a, y, b, length);
#elif __PARALLELTYPE__ == __MKL__
	cacu_saxpby_mkl(x, a, y, b, length);
#elif __PARALLELTYPE__ == __CUDA__
	cacu_saxpby_gpu(x, a, y, b, length);
#endif

}

/**
 * @cacu_scalex
 * math x[i] = a*x[i] :
 * x is a length dim array list, a is the corresponding scalar.
 */
inline void cacu_scalex(float *x, int length,const float a)
{

#if __PARALLELTYPE__ == __OPENBLAS__
	cacu_scalex_oblas(x, a, length);
#elif __PARALLELTYPE__ == __MKL__
	cacu_scalex_mkl(x, a, length);
#elif __PARALLELTYPE__ == __CUDA__
	cacu_scalex_gpu(x, a, length);
#endif

}

/**
 * @cacu_sgemv
 * math z = X*y:
 * trans_: whether x is needed to transpose
 */
inline void cacu_sgemv(TRANSPOSE trans_,const float *x, int x_height,const float *y, int x_width,const float alpha, float *z ,const float beta)
{

#if __PARALLELTYPE__ == __OPENBLAS__
	CBLAS_TRANSPOSE transx = (trans_ == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	cacu_sgemv_oblas(transx, x, x_height, y, x_width,alpha, z ,beta);
#elif __PARALLELTYPE__ == __MKL__
	CBLAS_TRANSPOSE transx = (trans_ == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	cacu_sgemv_mkl(transx, x, x_height, y, x_width,alpha, z ,beta);
#elif __PARALLELTYPE__ == __CUDA__
	cublasOperation_t transx = (trans_ == TRANS) ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N;
	cacu_sgemv_gpu(transx, x, x_height, y, x_width, alpha, z, beta);
#endif
}

/**
 * @cacu_sgemm
 * math z = a*X*Y + b*z:
 * transx_: whether x is need to transpose
 * transy_: whether y is need to transpose
 */
inline void cacu_sgemm(TRANSPOSE transx_, TRANSPOSE transy_,const float *x, int x_height, int x_width,const float *y, int y_width,const float alpha, float *z,const float beta)
{

#if __PARALLELTYPE__ == __OPENBLAS__
	CBLAS_TRANSPOSE transx = (transx_ == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	CBLAS_TRANSPOSE transy = (transy_ == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	cacu_sgemm_oblas(transx, transy, x, x_height, x_width, y, y_width, alpha, z, beta);
#elif __PARALLELTYPE__ == __MKL__
	CBLAS_TRANSPOSE transx = (transx_ == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	CBLAS_TRANSPOSE transy = (transy_ == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	cacu_sgemm_mkl(transx, transy, x, x_height, x_width, y, y_width, alpha, z, beta);
#elif __PARALLELTYPE__ == __CUDA__
	cublasOperation_t transx = (transx_ == TRANS) ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N;
	cublasOperation_t transy = (transy_ == TRANS) ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N;
	cacu_sgemm_gpu(transx, transy, x, x_height, x_width, y, y_width, alpha, z, beta);
#endif
}

/**
 * @cacu_copy
 * math y = x:
 * length: the input data's size
 */
inline void cacu_copy(const float_t *x, int length, float_t *y)
{
#if __PARALLELTYPE__ == __CUDA__
	cacu_copy_gpu(x, length, y);
#else
	memcpy(y,x,length*sizeof(float_t));
#endif
}


/**
 * @cacu_saxpy_atomic
 * math y = ax + y:
 * length: the input data's size
 */
inline void cacu_saxpy_atomic(const float_t *x, float_t a, float_t *y, int length)
{
#if __PARALLELTYPE__ == __CUDA__
	cacu_saxpy_atomic_gpu(x, a, y, length);
#elif __PARALLELTYPE__ == __OPENBLAS__
	cacu_saxpy_oblas(x, a, y, length);
#elif __PARALLELTYPE__ == __MKL__
	cacu_saxpy_mkl(x, a, y, length);
#endif
}

/**
 * @rand_vector
 * math vector_[i] = (rand()>=ratio_)
 * length: the input data's size
 */
inline void rand_vector(float_t *vector_, int length,const float_t ratio_)
{
	rand_t *rand = new rand_t();
#if __PARALLELTYPE__ == __CUDA__
	vec_t v_(length);

	for(int i = 0; i < length ; ++i)
	{
		if(rand->urand(0,1) >= ratio_)
			v_[i] = 1.0;
	}
	cuda_copy2dev(vector_, &v_[0], length);
	vec_t().swap(v_);
#else
	for(int i = 0; i < length ; ++i)
	{
		if(rand->urand(0,1) >= ratio_)
			vector_[i] = 1.0;
	}
#endif
	delete rand;
}


/**
 * @mask_vector
 * math vector_[i] = (vector_[i]>0)
 * length: the input data's size
 */
inline void mask_vector(float_t *vector_, int length, float_t *mask)
{
#if __PARALLELTYPE__ == __CUDA__
	vec_t v_(length);
	cuda_copy2host(&v_[0],vector_,length);
	for(int i = 0; i < length ; ++i)
	{
		if(v_[i] > 0)
			v_[i] = 1.0;
	}
	cuda_copy2dev(mask, &v_[0], length);
	vec_t().swap(v_);
#else
	for(int i = 0; i < length ; ++i)
	{
		if(vector_[i] > 0)
			mask[i] = 1.0;
	}
#endif
}


/**
 * @cacu_isaxdb
 * y[index] = x[index]*a + b
 */
inline void cacu_isaxb(const float_t *x, int length,const float_t a , const unsigned int *index_,const float_t b, float_t *y)
{
#if __PARALLELTYPE__ == __CUDA__
	cacu_isaxb_gpu(x,length, a, index_, b, y);
#else
	for (int i = 0 ; i < length ; ++i)
		y[i] = x[i];
	y[*index_] = a*x[*index_] + b;
#endif
}

inline unsigned int argmax(float_t *data, int length)
{

	unsigned int index;
#if __PARALLELTYPE__ == __CUDA__
	unsigned int *_index = cuda_malloc<unsigned int>(1,1);
	cacu_argmax_gpu(data,length,_index);
	cuda_copy2host(&index,_index,1);
	return index;
#else
	float_t max;
	unsigned int i;
	max = data[0];
	index = 0;
	for (i = 1 ; i < length; ++i)
	{
		if(data[i] > max){
			max = data[i];
			index = i;
		}
	}
	return index;
#endif
}




};
