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

#ifndef MATH_FUNCTIONS_CPU_H_
#define MATH_FUNCTIONS_CPU_H_

#include "../../definition.h"

#include "../math_definition.h"

namespace cacu {

void cacu_pow_cpu(float_t *x, const int length, const float_t pow, float_t *y);

void cacu_powt_cpu(float_t *x, const int length, const float_t *pow, float_t *y);

void cacu_ln_cpu(float_t *x, const int length, float_t *y);

void cacu_acos_cpu(float_t *x, const int length, float_t *y);

void cacu_cos_cpu(float_t *x, const int length, float_t *y);

/**
 * @rand_vector
 * math vector_[i] = (rand()>=ratio_)
 * length: the input data's size
 */
void rand_vector_cpu(float_t *vector_, const int length, const float_t ratio_);

/**
 * @mask_vector
 * math vector_[i] = (vector_[i]>0)
 * length: the input data's size
 */
void mask_vector_cpu(float_t *vector_, const int length, float_t *mask);

/**
* @mask_vector
* math vector_[i] = (vector_[i]<0)
* length: the input data's size
*/
void mask_vector_lt_cpu(float_t *vector_, const int length, float_t *mask);

/**
 * @cacu_copy
 * math y = x:
 * length: the input data's size
 */
void cacu_copy_cpu(float_t *x, const int length, float_t *y);

void cacu_abs_cpu(float_t *x, const int length, float_t *y);

void cacu_abs_grad_cpu(float_t *x, float_t *diff, const int length);

/**
 * @cacu_isaxdb
 * y[index] = x[index]*a + b
 */
void cacu_isaxb_cpu(float_t *x, const int channel, const int width, const int height, const float_t a, int *index_,
		const float_t b, float_t *y);

unsigned int argmax_cpu(float_t *data, const int length);

void cacu_transpose_cpu(float_t *mtx, const int m, const int n, const int clength);

void cacu_rotate_cpu(float_t *mtx, const int m, const int n, const int clength, rotate_code rotate);

void cacu_clip_vec_cpu(float_t *data, const float_t l_t, const float_t r_t, const int length);

void cacu_flip_cpu(int *x, const int length);

void cacu_floor_cpu(float_t *x, const int length, float_t *	y);

}

#endif /* MATH_FUNCTIONS_CPU_H_ */
