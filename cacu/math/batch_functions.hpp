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

#include "cuda/batch_functions_cuda.h"
#include "cpu/batch_functions_cpu.h"

namespace cacu {

/**
 * @cacu_sumbysize
 * sum by size:
 * accumulate the value by width or height , width is the matrix array's width dim which stored in row -major format.
 * sum by width y is (length/ width) height dim, sum by height y is width dim.
 * warning: take seriously this function may create accumulated error when width is large enough
 */
template<typename DTYPE>
inline void cacu_sumbysize(SUM SUMTYPE, const DTYPE *x, int length,
		const float_t alpha, DTYPE *y, const float_t beta, int width) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_sumbysize_cuda(SUMTYPE, x, length, alpha, y, beta, width);
#endif
#else
	cacu_sumbysize_cpu(SUMTYPE,x,length,alpha,y,beta,width);
#endif
}

/**
 * @cacu_cxsize
 * math y[i] = a[j]*x[i] :
 * x is a length dim array list, a is a size dim array list, a[j] is the corresponding scalar, j = i / (length / size).
 */
template<typename DTYPE>
inline void cacu_cxsize(const DTYPE *x, int length, const DTYPE *a, int size,
		DTYPE *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_cxsize_cuda(x, length, a, size, y);
#endif
#else
	cacu_cxsize_cpu(x, length, a, size,y);
#endif
}

/**
 * @cacu_cdxsize
 * math y[i] = x[i] / a[j] :
 * x is a length dim array list, a is a size dim array list, a[j] is the corresponding denominator, j = i / (length / size).
 */
template<typename DTYPE>
inline void cacu_cdxsize(const DTYPE *x, int length, const DTYPE *a, int size,
		DTYPE *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_cdxsize_cuda(x, length, a, size, y);
#endif
#else
	cacu_cdxsize_cpu(x, length, a, size, y);
#endif
}

/**
 * @cacu_ssxpy
 * math z[i] = a * x[j] + b * y[i] :
 * y is a length dim array list, x is a size dim array list, x[j] is the corresponding scalar, j = i / (length / size).
 * a & b are corresponding scalars for x, y
 */
template<typename DTYPE>
inline void cacu_ssxpy(const DTYPE *x, const DTYPE a, int size, const DTYPE *y,
		const DTYPE b, int length, DTYPE *z) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_ssxpy_cuda(x, a, size, y, b, length, z);
#endif
#else
	cacu_ssxpy_cpu(x, a, size, y, b, length, z);
#endif
}

/**
 * @cacu_sqr
 * math y[i] = x[i]^2 :
 */
template<typename DTYPE>
inline void cacu_sqr(const DTYPE *x, int length, DTYPE *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_sqr_cuda(x, length, y);
#endif
#else
	cacu_sqr_cpu(x,length,y);
#endif
}

/**
 * @cacu_root
 * math y[i] = sqrt(x[i]) :
 */
template<typename DTYPE>
inline void cacu_root(const DTYPE *x, int length, DTYPE *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_root_cuda(x, length, y);
#endif
#else
	cacu_root_cpu(x,length,y);
#endif
}

/**
 * @cacu_stdbychannel
 * math std[i] = sqrt(varience[i] + epsilon) :
 */
template<typename DTYPE>
inline void cacu_stdbychannel(const DTYPE *varience, int length, DTYPE *std,
		const DTYPE epsilon) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_stdbychannel_cuda(varience, length, std, epsilon);
#endif
#else
	cacu_stdbychannel_cpu(varience,length,std,epsilon);
#endif
}

/**
 * @cacu_bn_rou_grad
 * calculate the gradient of bn layer's rou
 * x: input feature
 * d_x: gradient of ^x
 * mean: mean of batch
 * std: standard deviation of batch
 * length: size of a feature map
 * d_rou: gradient of batch's variance
 */
template<typename DTYPE>
inline void cacu_bn_rou_grad(const DTYPE *x, const DTYPE *d_x,
		const DTYPE *mean, const DTYPE *std, int num, int length, int channel,
		DTYPE *d_rou) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_bn_rou_grad_cuda(x, d_x, mean, std, num, length, channel, d_rou);
#endif
#else
	cacu_bn_rou_grad_cpu(x, d_x, mean, std, num, length, channel, d_rou);
#endif
}

/**
 * @cacu_bn_mu_grad
 * calculate the gradient of bn layer's mu
 * x: input feature
 * d_x: gradient of ^x
 * mean: mean of batch
 * std: standard deviation of batch
 * d_rou: gradient of batch's variance
 * length: size of a feature map
 * d_mean: gradient of batch's mean
 */
template<typename DTYPE>
inline void cacu_bn_mu_grad(const DTYPE *x, const DTYPE *d_x, const DTYPE *mean,
		const DTYPE *std, const DTYPE *d_rou, int num, int length, int channel,
		DTYPE *d_mean) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_bn_mu_grad_cuda(x, d_x, mean, std, d_rou, num, length, channel, d_mean);
#endif
#else
	cacu_bn_mu_grad_cpu(x, d_x, mean, std, d_rou, num, length ,channel, d_mean);
#endif
}

/**
 * @cacu_bn_dx_grad
 * calculate the gradient of bn layer's dx
 * x: input feature
 * d_x: gradient of ^x
 * mean: mean of batch
 * std: standard deviation of batch
 * d_rou: gradient of batch's variance
 * d_mean: gradient of batch's mean
 * length: size of a feature map
 * dx: gradient of x
 */
template<typename DTYPE>
inline void cacu_bn_dx_grad(const DTYPE *x, const DTYPE *d_x, const DTYPE *mean,
		const DTYPE *std, const DTYPE *d_rou, const DTYPE *d_mean, int num,
		int length, int channel, DTYPE *dx) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_bn_dx_grad_cuda(x, d_x, mean, std, d_rou, d_mean, num, length, channel,
			dx);
#endif
#else
	cacu_bn_dx_grad_cpu(x, d_x, mean, std, d_rou, d_mean, num, length, channel, dx);
#endif
}

/**
 * @cacu_bn_gamma_grad
 * calculate the gradient of bn layer's scale
 * _x: is ^x
 * d_y: gradient propagate form top layer
 * length: size of a feature map
 * d_gamma: gradient of gamma
 */
template<typename DTYPE>
inline void cacu_bn_gamma_grad(const DTYPE *_x, const DTYPE *d_y, int num,
		int length, int channel, DTYPE *d_gamma) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_bn_gamma_grad_cuda(_x, d_y, num, length, channel, d_gamma);
#endif
#else
	cacu_bn_gamma_grad_cpu(_x, d_y, num, length, channel, d_gamma);
#endif
}

/**
 * @cacu_ssx
 * math y[i] *= x[i] :
 * scale by element wise.
 */
template<typename DTYPE>
inline void cacu_ssx(const DTYPE *x, int length, DTYPE *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_ssx_cuda(x, length, y);
#endif
#else
	cacu_ssx_cpu(x, length, y);
#endif
}

/**
 * @cacu_group_alloc
 * alloc data by group
 */
template<typename DTYPE>
inline void cacu_group_alloc(int num, int channel, int channel_length,
		int group, DTYPE *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_group_alloc_cuda(num, channel, channel_length, group, y);
#endif
#else
	cacu_group_alloc_cpu(num, channel, channel_length, group, y);
#endif
}

/**
 * @cacu_group_combine
 * combine data by group
 */
template<typename DTYPE>
inline void cacu_group_combine(int num, int channel, int channel_length,
		int group, DTYPE *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_group_combine_cuda(num, channel, channel_length, group, y);
#endif
#else
	cacu_group_combine_cpu(num, channel, channel_length, group, y);
#endif
}

}
;
