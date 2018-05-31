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

#ifndef ACTIVITION_FUNCTIONS_CPU_H_
#define ACTIVITION_FUNCTIONS_CPU_H_

#include <algorithm>
#include "../../definition.h"

namespace cacu {

/**
 * @cacu_relu
 * math x[i] = max(0,x[i]) :
 * for activation use relu functions.
 */
void cacu_relu_cpu(float_t *x, const int length);

/**
 * @cacu_relu_grad
 * math if(x[i]<0)?g[i] = g[i]:g[i] = 0;
 * gradient for activation use relu functions.
 */
void cacu_relu_grad_cpu(const float_t *x, float_t *g, const int length);

/**
 * @cacu_leaky_relu
 * math if(x[i]<0)?x[i] = x[i]:x[i] *= a;
 * for activation use leaky_relu functions.
 */
void cacu_leaky_relu_cpu(float_t *x, const float_t a, const int length);

/**
 * @cacu_leaky_relu_grad
 * math if(x[i]<0)?g[i] = g[i]:g[i] *= a;
 * gradient for activation use leaky_relu functions.
 */
void cacu_leaky_relu_grad_cpu(const float_t *x, float_t *g, const float_t a,
		const int length);

void cacu_prelu_cpu(float_t *x, const float_t *slopes,
		const int num, const int channel, const int c_length);

void cacu_prelu_grad_cpu(const float_t *x, float_t *g,
		const float_t *slopes, float_t * g_slopes, const int num,
		const int channel, const int c_length);


/**
 * @cacu_softmax
 * math softmax;
 * for activation use softmax functions.
 */
void cacu_softmax_cpu(const float_t *x, const int num, const int channel,
		const int width, const int height, float_t *y);

/**
 * @cacu_tanh
 * math tanh;
 * for activation use tanh functions.
 */
void cacu_tanh_cpu(const float_t *x, const int length, float_t *y);

/**
 * @cacu_tanh_grad
 * math tanh;
 * for activation use tanh functions.
 */
void cacu_tanh_grad_cpu(const float_t *x, const float_t *g, const int length, float_t *y);

/**
* @cacu_htanh
* math half tanh;
* for activation use tanh functions.
*/
void cacu_htanh_cpu(const float_t *x, const int length, float_t *y);

/**
* @cacu_htanh_grad
* math half tanh;
* for activation use tanh functions.
*/
void cacu_htanh_grad_cpu(const float_t *x, const float_t *g, const int length, float_t *y);

/**
 * @cacu_tanh
 * math tanh;
 * for activation use tanh functions.
 */
void cacu_sigmoid_cpu(const float_t *x, const int length, float_t *y);

/**
 * @cacu_sigmoid_grad
 * math sigmoid;
 * for activation use sigmoid functions.
 */
void cacu_sigmoid_grad_cpu(const float_t *x, const float_t *g, const int length,
		float_t *y);

float_t sigmoid(float_t data);

float_t tanh(float_t data);

}

#endif
