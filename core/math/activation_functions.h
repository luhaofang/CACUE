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


#include "cuda/activation_functions_cuda.h"

namespace mycnn{

	/**
	 * @cacu_relu
	 * math x[i] = max(0,x[i]) :
	 * for activation use relu functions.
	 */
	inline void cacu_relu(float_t *x, int length)
	{

#if __PARALLELTYPE__ == __GPU__
		cacu_relu_gpu(x,length);
#else
		int i;
#if __OPENMP__ == ON
		#pragma omp parallel for default(shared) private(i)
#endif
		for (i = 0; i < length; ++i)
		{
			if (x[i] < 0)
				x[i] = 0.0;
		}
#endif

	}

	/**
	 * @cacu_relu_grad
	 * math if(x[i]<0)?g[i] = g[i]:g[i] = 0;
	 * gradient for activation use relu functions.
	 */
	inline void cacu_relu_grad(float_t *x, float_t *g, int length)
	{

#if __PARALLELTYPE__ == __GPU__
		cacu_relu_grad_gpu(x,g,length);
#else
		int i;
#if __OPENMP__ == ON
		#pragma omp parallel for default(shared) private(i)
#endif
		for (i = 0; i < length; ++i)
		{
			if (x[i] <= 0)
				g[i] = 0.0;
		}
#endif

	}

	/**
	 * @cacu_leaky_relu
	 * math if(x[i]<0)?x[i] = x[i]:x[i] *= a;
	 * for activation use leaky_relu functions.
	 */
	inline void cacu_leaky_relu(float_t *x, float_t a, int length)
	{

#if __PARALLELTYPE__ == __GPU__
		cacu_leaky_relu_gpu(x, a, length);
#else
		int i;
#if __OPENMP__ == ON
		#pragma omp parallel for default(shared) private(i)
#endif
		for (i = 0; i < length; ++i)
		{
			if (x[i] < 0)
				x[i] *= a;
		}
#endif

	}

	/**
	 * @cacu_leaky_relu_grad
	 * math if(x[i]<0)?g[i] = g[i]:g[i] *= a;
	 * gradient for activation use leaky_relu functions.
	 */
	inline void cacu_leaky_relu_grad(float_t *x,float_t *g, float_t a, int length)
	{

#if __PARALLELTYPE__ == __GPU__
		cacu_leaky_relu_grad_gpu(x, g, a, length);
#else
		int i;
#if __OPENMP__ == ON
		#pragma omp parallel for default(shared) private(i)
#endif
		for (i = 0; i < length; ++i)
		{
			if (x[i] <= 0)
				g[i] *= a;
		}
#endif

	}

	/**
	 * @cacu_softmax
	 * math softmax;
	 * for activation use softmax functions.
	 */
	inline void cacu_softmax(float_t *x, int num, int length,float_t *y)
	{

#if __PARALLELTYPE__ == __GPU__
		cacu_softmax_gpu(x, num, length, y);
#else
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
#endif

	}

};
