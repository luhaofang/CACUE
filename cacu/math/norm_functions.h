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

#ifndef NORM_FUNCTIONS_H_
#define NORM_FUNCTIONS_H_

#include "../definition.h"

#include "cuda/norm_functions_cuda.h"
#include "cpu/norm_functions_cpu.h"

namespace cacu {

/**
 * @cacu_norm_l1
 * math x[i] = x[i] / sum(abs(x)) :
 * for l1 norm functions.
 */
inline void cacu_l1_normalization(float_t *x, const int length, const float_t epsilon) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_normalization_l1_cuda(x, length, epsilon);
#endif
#else
	cacu_normalization_l1_cpu(x, length, epsilon);
#endif
}

/**
 * @cacu_norm_l1_grad
 * math x[i] = x[i] / sum(abs(x)) :
 * for l1 norm grad functions.
 */
inline void cacu_l1_normalization_grad(float_t *x, const int length, const float_t epsilon, float_t *yg, float_t *xg) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_normalization_l1_grad_cuda(x, length, epsilon, yg, xg);
#endif
#else
	cacu_normalization_l1_grad_cpu(x, length, epsilon, yg, xg);
#endif
}

/**
 * @cacu_norm_l2
 * math x[i] = x[i] / sqrt(sum((x*x))) :
 * for l2 norm functions
 */
inline void cacu_l2_normalization(float_t *x, const int length,  const float_t epsilon) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_normalization_l2_cuda(x, length, epsilon);
#endif
#else
	cacu_normalization_l2_cpu(x, length, epsilon);
#endif
}

/**
 * @cacu_norm_l2_ grad
 * math x[i] = x[i] / sqrt(sum((x*x))) :
 * for l2 norm grad functions
 */
inline void cacu_l2_normalization_grad(float_t *x, const int length,  const float_t epsilon, float_t *yg, float_t *xg) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_normalization_l2_grad_cuda(x, length, epsilon, yg, xg);
#endif
#else
	cacu_normalization_l2_grad_cpu(x, length, epsilon, yg, xg);
#endif
}

/*
 * @cacu_norm
 * math x[i] =
 * compute the norm of the tenosr data.
 * norm_l1, norm_l2
 */
inline void cacu_norm(float_t *x, const int length, const float_t epsilon, norm_type type, float_t* norm)
{
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	switch(type)
	{
		case norm_l1:
			cacu_norm_l1_cuda(x, length, epsilon, norm);
			break;
		case norm_l2:
			cacu_norm_l2_cuda(x, length, epsilon, norm);
			break;
		default:
			break;
	}
#endif
#else
	switch(type)
	{
		case norm_l1:
			cacu_norm_l1_cpu(x, length, epsilon, norm);
			break;
		case norm_l2:
			cacu_norm_l2_cpu(x, length, epsilon, norm);
			break;
		default:
			break;
	}
#endif
}



}


#endif
