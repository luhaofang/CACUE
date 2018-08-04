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

#include "activation_functions_cpu.h"

namespace cacu {

/**
 * @cacu_relu
 * math x[i] = max(0,x[i]) :
 * for activation use relu functions.
 */
void cacu_relu_cpu(float_t *x, const int length) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		x[i] = max(x[i], float_t(0));
	}
}

/**
 * @cacu_relu_grad
 * math if(x[i]<0)?g[i] = g[i]:g[i] = 0;
 * gradient for activation use relu functions.
 */
void cacu_relu_grad_cpu(float_t *x, float_t *g, const int length) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		//if (x[i] <= 0)
		g[i] *= (x[i] > 0);
	}
}

/**
 * @cacu_leaky_relu
 * math if(x[i]<0)?x[i] = x[i]:x[i] *= a;
 * for activation use leaky_relu functions.
 */
void cacu_leaky_relu_cpu(float_t *x, const float_t a, const int length) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		if (x[i] < 0)
			x[i] *= a;
	}
}

/**
 * @cacu_leaky_relu_grad
 * math if(x[i]<0)?g[i] = g[i]:g[i] *= a;
 * gradient for activation use leaky_relu functions.
 */
void cacu_leaky_relu_grad_cpu(float_t *x, float_t *g, const float_t a,
		const int length) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		if (x[i] <= 0)
			g[i] *= a;
	}
}

/**
 * @cacu_leaky_relu
 * math if(x[i]<0)?x[i] = x[i]:x[i] *= a;
 * for activation use leaky_relu functions.
 */
void cacu_prelu_cpu(float_t *x, const float_t *slopes, const int num,
		const int channel, const int c_length) {
	int i;
	int length = num * channel * c_length;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		x[i] = x[i] * slopes[i / c_length % channel] * (x[i] <= 0)
			+ x[i] * (x[i] > 0);
	}
}

/**
 * @cacu_leaky_relu_grad
 * math if(x[i]<0)?g[i] = g[i]:g[i] *= a;
 * gradient for activation use leaky_relu functions.
 */
void cacu_prelu_grad_cpu(float_t *x, float_t *g, const float_t *slopes,
		float_t * g_slopes, const int num, const int channel,
		const int c_length) {
	int i;
	int length = num * channel * c_length;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		g[i] = g[i] * slopes[i / c_length % channel] * (x[i] <= 0)
			+ g[i] * (x[i] > 0);
		g_slopes[i / c_length % channel] += g[i] * x[i] * (x[i] <= 0);
	}
}

/**
 * @cacu_softmax
 * math softmax;
 * for activation use softmax functions.
 */
void cacu_softmax_cpu(float_t *x, const int num, const int channel,
		const int width, const int height, float_t *y) {
	float_t *xp, *yp, max_, sum_;
	int n, i;
	int length = width * height * num;
	int c_length = width * height;
	int p_length = channel * c_length;
	int index;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(n,i,max_,sum_,xp,yp)
#endif
	for (n = 0; n < length; ++n) {
		index = n / c_length * p_length + n % c_length;
		xp = x + index;
		yp = y + index;
		max_ = xp[0];
		sum_ = 0;
		for (i = 1; i < channel; ++i)
			max_ = max(xp[i * c_length], max_);
		for (i = 0; i < channel; ++i) {
			yp[i * c_length] = exp(xp[i * c_length] - max_);
			sum_ += yp[i * c_length];
		}
		for (i = 0; i < channel; ++i) {
			yp[i * c_length] /= sum_;
		}
	}
}


/**
 * @cacu_elu
 * math y[i] = x[i] = (max(x[i], float_t(0))
		        + alpha * (exp(min(x[i], float_t(0))) - float_t(1)));
 * for activation use elu functions.
 */
void cacu_elu_cpu(float_t *x, const int length, const float_t alpha, float_t *y) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		y[i] = (max(x[i], float_t(0))
				        + alpha * (exp(min(x[i], float_t(0))) - float_t(1)));
	}
}

/**
 * @cacu_elu_grad
 * math g[i] = (grad[i] * ((x[i] > 0)
		          + (alpha + y[i]) * (x[i] <= 0)))
 * gradient for activation use elu functions.
 */
void cacu_elu_grad_cpu(float_t *x, float_t *g, const int length, const float_t alpha, float_t *y, float_t *grad) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		//if (x[i] <= 0)
		g[i] = (grad[i] * ((x[i] > 0)
		          + (alpha + y[i]) * (x[i] <= 0)));

	}
}

/**
 * @cacu_exp
 * math exp;
 * for activation use exp functions.
 */
void cacu_exp_cpu(float_t *x, const int length, float_t *y) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		y[i] = exp(x[i]);
	}
}

/**
 * @cacu_tanh
 * math tanh;
 * for activation use tanh functions.
 */
void cacu_tanh_cpu(float_t *x, const int length, float_t *y) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		y[i] = tanh(x[i]);
	}
}

/**
 * @cacu_tanh_grad
 * math tanh;
 * for activation use tanh functions.
 */
void cacu_tanh_grad_cpu(float_t *x, float_t *g, const int length, float_t *y) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		y[i] = g[i] * (float_t(1) - x[i] * x[i]);
	}
}

/**
* @cacu_htanh
* math half tanh;
* for activation use tanh functions.
*/
void cacu_htanh_cpu(float_t *x, const int length, float_t *y) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		y[i] = x[i] > 0 ? tanh(x[i]) : 0;
	}
}

/**
* @cacu_htanh_grad
* math half tanh;
* for activation use tanh functions.
*/
void cacu_htanh_grad_cpu(float_t *x, float_t *g, const int length, float_t *y) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		y[i] = x[i] > 0 ? g[i] * (float_t(1) - x[i] * x[i]) : 0;
	}
}


/**
 * @cacu_tanh
 * math tanh;
 * for activation use tanh functions.
 */
void cacu_sigmoid_cpu(float_t *x, const int length, float_t *y) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		y[i] = float_t(1) / (float_t(1) + exp(-x[i]));
	}
}

/**
 * @cacu_sigmoid_grad
 * math sigmoid;
 * for activation use sigmoid functions.
 */
void cacu_sigmoid_grad_cpu(float_t *x, float_t *g, const int length,
		float_t *y) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i) {
		y[i] = g[i] * x[i] * (float_t(1) - x[i]);
	}
}

float_t sigmoid(float_t data) {
	return float_t(1) / (float_t(1) + exp(-data));
}

float_t tanh(float_t data) {
	return float_t(2) * sigmoid(float_t(2) * data) - float_t(1);
}

}
