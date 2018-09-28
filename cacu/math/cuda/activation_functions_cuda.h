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

#ifndef ACTIVATION_FUNCTIONS_CUDA_H_
#define ACTIVATION_FUNCTIONS_CUDA_H_

#include "../../config.h"

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__

#include "../../definition.h"

namespace cacu {

/**
 * for activation use relu functions in cuda
 */
extern "C" void cacu_relu_cuda(float_t *x, int length);

/**
 * gradient for activation use relu functions in cuda
 */
extern "C" void cacu_relu_grad_cuda(float_t *x, float_t *g, int length);

/**
 * for activation use leaky_relu functions in cuda
 */
extern "C" void cacu_leaky_relu_cuda(float_t *x, float_t a, int length);

/**
 * gradient for activation use leaky_relu functions in cuda
 */
extern "C" void cacu_leaky_relu_grad_cuda(float_t *x, float_t *g, float_t a,
		int length);

extern "C" void cacu_prelu_cuda(float_t *x, const float_t *slopes,
		const int num, const int channel, const int c_length);

extern "C" void cacu_prelu_grad_cuda(float_t *x, float_t *g,
		const float_t *slopes, float_t * g_slopes, const int num,
		const int channel, const int c_length);

/**
 * for activation use softmax functions in cuda
 */
extern "C" void cacu_softmax_cuda(float_t *x, const int num, const int channel,
		const int width, const int height, float_t *y);

/**
 * for activation use tanh functions in cuda
 */
extern "C" void cacu_exp_cuda(float_t *x, int length, float_t *y);

/**
 * for activation use selu functions in cuda
 */
extern "C" void cacu_elu_cuda(float_t *x, const int length, const float_t alpha, float_t *y);

/**
 * gradient for activation use selu functions in cuda
 */
extern "C" void cacu_elu_grad_cuda(float_t *x, float_t *g, const int length, const float_t alpha, float_t *y, float_t *grad);

/**
 * for activation use tanh functions in cuda
 */
extern "C" void cacu_tanh_cuda(float_t *x, int length, float_t *y);

/**
 * gradient for activation use tanh functions in cuda
 */
extern "C" void cacu_tanh_grad_cuda(float_t *x, float_t *g, int length,
		float_t *y);

/**
 * for activation use half tanh functions in cuda
 */
extern "C" void cacu_htanh_cuda(float_t *x, const int length, float_t *y);

/**
* @cacu_htanh_grad
* math half tanh;
* for activation use half tanh functions.
*/
extern "C" void cacu_htanh_grad_cuda(float_t *x, float_t *g, const int length, float_t *y);

/**
 * for activation use sigmoid functions in cuda
 */
extern "C" void cacu_sigmoid_cuda(float_t *x, int length, float_t *y);

/**
 * gradient for activation use sigmoid functions in cuda
 */
extern "C" void cacu_sigmoid_grad_cuda(float_t *x, float_t *g, int length,
		float_t *y);

}

#endif
#endif

#endif

