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
#include "blas/math_function_oblas.h"
#include "blas/math_function_mkl.h"
#include "blas/math_function_cublas.h"
#include "utils/rand_t.h"

using namespace std;

namespace cacu {

/**
 * @cacu_saxpy
 * math y = a*x + y:
 * length: the input data's size
 */
inline void cacu_saxpy(float *x, const float a, float *y, const int length);

/**
 * @cacu_saxpby
 * math y = a*x + b*y:
 * length: the input data's size
 */
inline void cacu_saxpby(float *x, const float a, float *y, const float b,
		const int length);

/**
 * @cacu_scalex
 * math x[i] = a*x[i] :
 * x is a length dim array list, a is the corresponding scalar.
 */
inline void cacu_scalex(float *x, const int length, const float a);

/**
 * @cacu_sgemv
 * math z = a*X*y + b*z:
 * trans_: whether x is needed to transpose
 */
inline void cacu_sgemv(TRANSPOSE trans_, float *x, const int x_height,
		float *y, const int x_width, const float alpha, float *z,
		const float beta);

/**
 * @cacu_sgemm
 * math z = a*X*Y + b*z:
 * transx_: whether x is need to transpose
 * transy_: whether y is need to transpose((rand() % ( max - min + 1)) + min);
 */
inline void cacu_sgemm(TRANSPOSE transx_, TRANSPOSE transy_, float *x,
		const int x_height, const int x_width, float *y, const int y_width,
		const float alpha, float *z, const float beta);

/**
 * @cacu_saxpy_atomic
 * math y = ax + y:
 * length: the input data's size
 */
inline void cacu_saxpy_atomic(float *x, float a, float *y, const int length);

/**
 * @rand_vector
 * math vector_[i] = (rand()>=ratio_)
 * length: the input data's size
 */
inline void rand_vector(float *vector_, const int length, const float ratio_);

/**
 * @mask_vector
 * math vector_[i] = (vector_[i]>0)
 * length: the input data's size
 */
inline void mask_vector(float *vector_, const int length, float *mask);

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
inline void cacu_isaxb(float *x, const int length, const float a,
		unsigned int *index_, const float b, float *y);

inline unsigned int argmax(float *data, const int length);

inline void cacu_transpose(float *mtx, const int m, const int n);

inline void cacu_clip_vec(float *data, const float threshold, const int length);

}
