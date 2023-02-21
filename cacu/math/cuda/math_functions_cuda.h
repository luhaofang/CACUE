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

#ifndef MATH_FUNCTIONS_CUDA_H_
#define MATH_FUNCTIONS_CUDA_H_

#include "../../definition.h"
#include "../math_definition.h"

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__

namespace cacu{

extern "C" void cacu_saxpy_atomic_cuda(float_t *x, const float_t a, float_t *y, const int length);

/**
 * @cacu_isaxdb_cuda
 * y[index] = x[index]*a + b
 */
extern "C" void cacu_isaxb_cuda(float_t *x, const int channel, const int width, const int height, const float_t a ,int *index_,const float_t b, float_t *y);

extern "C" void cacu_argmax_cuda(float_t *x, const int length, unsigned int *index_);

extern "C" void cacu_transpose_cuda(float_t *mtx, const int m, const int n, const int clength);

extern "C" void cacu_clip_vec_cuda(float_t *data, const float_t l_t, const float_t r_t, const int length);

extern "C" void cacu_abs_cuda(float_t *x, const int length, float_t *y);

extern "C" void cacu_abs_grad_cuda(float_t *x, float_t *diff, const int length);

extern "C" void cacu_pow_cuda(float_t *x, const int length, const float_t pow, float_t *y);

extern "C" void cacu_powt_cuda(float_t *x, const int length, const float_t *pow, float_t *y);

extern "C" void cacu_ln_cuda(float_t *x, const int length, float_t *y);

extern "C" void cacu_flip_cuda(int *x, const int length);

extern "C" void rand_vector_cuda(float_t *vector_, const int length, const float_t ratio_);

extern "C" void cacu_acos_cuda(float_t *x, const int length, float_t *y);

extern "C" void cacu_cos_cuda(float_t *x, const int length, float_t *y);

extern "C" void cacu_floor_cuda(float_t *x, const int length, float_t *y);

extern "C" void cacu_rotate_cuda(float_t *mtx, const int m, const int n, const int clength, rotate_code rotate);

}

#endif
#endif


#endif
