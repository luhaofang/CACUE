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


#include "../../config.h"

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__

namespace cacu{

extern "C" void cacu_saxpy_atomic_cuda(float *x, const float a, float *y, const int length);

/**
 * @cacu_isaxdb_cuda
 * y[index] = x[index]*a + b
 */
extern "C" void cacu_isaxb_cuda(float *x, const int length, const float a ,int *index_,const float b, float *y);

extern "C" void cacu_argmax_cuda(float *x, const int length, unsigned int *index_);

extern "C" void cacu_transpose_cuda(float *mtx, const int m, const int n);

extern "C" void cacu_clip_vec_cuda(float *data, const float threshold, const int length);

extern "C" void cacu_abs_cuda(float *x, const int length, float *y);

extern "C" void cacu_abs_grad_cuda(float *x, float *diff, const int length, float *grad);

}

#endif
#endif


#endif
