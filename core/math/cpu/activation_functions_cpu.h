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

#include "../../utils/data_defination.h"


namespace mycnn{

	/**
	 * @cacu_relu
	 * math x[i] = max(0,x[i]) :
	 * for activation use relu functions.
	 */
	inline void cacu_relu_cpu(float_t *x, int length)
	{
		int i;
#if __OPENMP__ == ON
		#pragma omp parallel for default(shared) private(i)
#endif
		for (i = 0; i < length; ++i)
		{
			x[i] = max(x[i], float_t(0));
		}
	}

	/**
	 * @cacu_relu_grad
	 * math if(x[i]<0)?g[i] = g[i]:g[i] = 0;
	 * gradient for activation use relu functions.
	 */
	inline void cacu_relu_grad_cpu(float_t *x, float_t *g, int length)
	{
		int i;
#if __OPENMP__ == ON
		#pragma omp parallel for default(shared) private(i)
#endif
		for (i = 0; i < length; ++i)
		{
			if (x[i] <= 0)
				g[i] = float_t(0);
		}
	}

	/**
	 * @cacu_leaky_relu
	 * math if(x[i]<0)?x[i] = x[i]:x[i] *= a;
	 * for activation use leaky_relu functions.
	 */
	inline void cacu_leaky_relu_cpu(float_t *x, float_t a, int length)
	{
		int i;
#if __OPENMP__ == ON
		#pragma omp parallel for default(shared) private(i)
#endif
		for (i = 0; i < length; ++i)
		{
			if (x[i] < 0)
				x[i] *= a;
		}
	}

	/**
	 * @cacu_leaky_relu_grad
	 * math if(x[i]<0)?g[i] = g[i]:g[i] *= a;
	 * gradient for activation use leaky_relu functions.
	 */
	inline void cacu_leaky_relu_grad_cpu(float_t *x,float_t *g, float_t a, int length)
	{
		int i;
#if __OPENMP__ == ON
		#pragma omp parallel for default(shared) private(i)
#endif
		for (i = 0; i < length; ++i)
		{
			if (x[i] <= 0)
				g[i] *= a;
		}
	}

	/**
	 * @cacu_softmax
	 * math softmax;
	 * for activation use softmax functions.
	 */
	inline void cacu_softmax_cpu(float_t *x, int num, int length,float_t *y)
	{
		float_t *xp,*yp,max_,sum_;
		int n,i;
#if __OPENMP__ == ON
		#pragma omp parallel for default(shared) private(n,i,max_,sum_,xp,yp)
#endif
		for (n = 0; n < num; ++n)
		{
			xp = x + n * length;
			yp = y + n * length;
			max_ = xp[0];
			sum_ = 0;
			for (i = 1; i < length; ++i)
				max_ = max(xp[i], max_);
			for (i = 0; i < length; ++i)
			{
				yp[i] = exp(xp[i] - max_);
				sum_ += yp[i];
			}
			for (i = 0; i < length; ++i)
			{
				yp[i] = (yp[i] / sum_);
			}
		}
	}

	/**
	 * @cacu_tanh
	 * math tanh;
	 * for activation use tanh functions.
	 */
	inline void cacu_tanh_cpu(float_t *x, int length,float_t *y)
	{
		int i;
#if __OPENMP__ == ON
		#pragma omp parallel for default(shared) private(i)
#endif
		for (i = 0; i < length; ++i)
		{
			y[i] = tanh(x[i]);
		}
	}

	/**
	 * @cacu_tanh_grad
	 * math tanh;
	 * for activation use tanh functions.
	 */
	inline void cacu_tanh_grad_cpu(float_t *x, float_t *g, int length, float_t *y)
	{
		int i;
#if __OPENMP__ == ON
		#pragma omp parallel for default(shared) private(i)
#endif
		for (i = 0; i < length; ++i)
		{
			y[i] = g[i] * (float_t(1) - x[i] * x[i]);
		}
	}

	/**
	 * @cacu_tanh
	 * math tanh;
	 * for activation use tanh functions.
	 */
	inline void cacu_sigmoid_cpu(float_t *x, int length,float_t *y)
	{
		int i;
#if __OPENMP__ == ON
		#pragma omp parallel for default(shared) private(i)
#endif
		for (i = 0; i < length; ++i)
		{
			y[i] = float_t(1)/(float_t(1) + exp(-x[i]));
		}
	}

	/**
	 * @cacu_sigmoid_grad
	 * math sigmoid;
	 * for activation use sigmoid functions.
	 */
	inline void cacu_sigmoid_grad_cpu(float_t *x, float_t *g, int length, float_t *y)
	{
		int i;
#if __OPENMP__ == ON
		#pragma omp parallel for default(shared) private(i)
#endif
		for (i = 0; i < length; ++i)
		{
			y[i] = g[i] * x[i] * (float_t(1) - x[i]);
		}
	}

	template<typename DTYPE>
	inline DTYPE sigmoid(DTYPE data)
	{
		return float_t(1) / (float_t(1) + exp(-data));
	}

	template <typename Dtype>
	inline Dtype tanh(Dtype data)
	{
		return float_t(2) * sigmoid(float_t(2) * data) - float_t(1);
	}


};
