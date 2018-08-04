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

#ifndef BATCH_FUNCTIONS_CPU_H_
#define BATCH_FUNCTIONS_CPU_H_

#include <math.h>
#include "../../definition.h"

namespace cacu {

/**
 * @cacu_sumbysize
 * sum by size:
 * accumulate the value by width or height , width is the matrix array's width dim which stored in row -major format.
 * sum by width y is (length/ width) height dim, sum by height y is width dim.
 * warning: take seriously this function may create accumulated error when width is large enough
 */
template<typename DTYPE>
void cacu_sumbysize_cpu(SUM SUMTYPE, DTYPE *x, int length,
		const float_t alpha, DTYPE *y, const float_t beta, int width) {
	int height = length / width;
	int b, i;
	DTYPE acc;
	if (BYWIDTH == SUMTYPE) {
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(b,i,acc,xp,alpha,beta)
#endif
		for (b = 0; b < height; ++b) {
			acc = DTYPE(0);
			for (i = 0; i < width; ++i)
				acc += x[i + b * width];
			y[b] = ((alpha * acc) + beta * y[b]);
		}
	} else if (BYHEIGHT == SUMTYPE) {
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(b,i,acc,xp,alpha,beta)
#endif
		for (b = 0; b < width; ++b) {
			acc = DTYPE(0);
			for (i = 0; i < height; ++i)
				acc += x[i * width + b];
			y[b] = ((alpha * acc) + beta * y[b]);
		}
	}
}

/**
 * @cacu_cxsize
 * math y[i] = a[j]*x[i] :
 * x is a length dim array list, a is a size dim array list, a[j] is the corresponding scalar, j = i / (length / size).
 */
template<typename DTYPE>
void cacu_cxsize_cpu(DTYPE *x, int length, DTYPE *a,
		int size, DTYPE *y) {
	int block_size = length / size;
	DTYPE *yp;
	int b, j;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(b,j,yp,xp)
#endif
	for (b = 0; b < size; ++b) {
		yp = y + b * block_size;
		for (j = 0; j < block_size; ++j)
			yp[j] = x[j + b * block_size] * a[b];
	}
}

/**
 * @cacu_cdxsize
 * math y[i] = x[i] / a[j] :
 * x is a length dim array list, a is a size dim array list, a[j] is the corresponding denominator, j = i / (length / size).
 */
template<typename DTYPE>
void cacu_cdxsize_cpu(DTYPE *x, int length, DTYPE *a,
		int size, DTYPE *y) {
	int block_size = length / size;
	DTYPE *yp;
	int b, j;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(b,j,yp,xp)
#endif
	for (b = 0; b < size; ++b) {
		yp = y + b * block_size;
		for (j = 0; j < block_size; ++j)
			yp[j] = x[j + b * block_size] / a[b];
	}
}

/**
 * @cacu_sdxsize
 * math y[i] = (x[i] + a) / b:
 * x is a length dim array list, a is the corresponding denominator.
 */
template<typename DTYPE>
void cacu_sdxsize_cpu(DTYPE *x, int length, DTYPE a, DTYPE b, DTYPE *y) {
	int j;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(j)
#endif
	for (j = 0; j < length; ++j)
		y[j] = (x[j] + a) / b;
}

/**
 * @cacu_ssxpy
 * math z[i] = a * x[j] + b * y[i] :
 * y is a length dim array list, x is a size dim array list, x[j] is the corresponding scalar, j = i / (length / size).
 * a & b are corresponding scalars for x, y
 */
template<typename DTYPE>
void cacu_ssxpy_cpu(DTYPE *x, const DTYPE a, int size,
		DTYPE *y, const DTYPE b, int length, DTYPE *z) {
	int block_size = length / size;
	DTYPE *zp;
	int i, j;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i,j,yp,zp)
#endif
	for (i = 0; i < size; ++i) {
		zp = z + i * block_size;
		for (j = 0; j < block_size; ++j)
			zp[j] = a * x[i] + b * y[j + i * block_size];
	}
}

/**
 * @cacu_sqr
 * math y[i] = x[i]^2 :
 */
template<typename DTYPE>
void cacu_sqr_cpu(DTYPE *x, int length, DTYPE *y) {
	int j;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(j)
#endif
	for (j = 0; j < length; ++j)
		y[j] = x[j] * x[j];
}

/**
 * @cacu_root
 * math y[i] = sqrt(x[i]) :
 */
template<typename DTYPE>
void cacu_root_cpu(DTYPE *x, int length, DTYPE *y) {

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(j)
#endif
	for (int j = 0; j < length; ++j)
		y[j] = sqrtf((float_t) x[j]);
}

/**
 * @cacu_stdbychannel
 * math std[i] = sqrt(varience[i] + epsilon) :
 */
template<typename DTYPE>
void cacu_stdbychannel_cpu(DTYPE *varience, int length, DTYPE *std,
		const DTYPE epsilon) {
	int j;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(j)
#endif
	for (j = 0; j < length; ++j)
		std[j] = (float_t) pow(varience[j] + epsilon, 0.5);
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
void cacu_bn_rou_grad_cpu(const DTYPE *x, const DTYPE *d_x,
		const DTYPE *mean, const DTYPE *std, int num, int length, int channel,
		DTYPE *d_rou) {
	int cin_length = length / channel;

	int set;

	int n, c, f;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(n,c,f,set)
#endif
	for (n = 0; n < num; n++) {
		//iteration for channel
		for (c = 0; c < channel; c++) {
			set = n * length + cin_length * c;
			//iteration for feature map
			for (f = 0; f < cin_length; f++) {

				d_rou[c] += (d_x[set + f] * (x[set + f] - mean[c])
						* (DTYPE(-0.5) / pow(std[c], 3)));
			}
		}
	}
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
void cacu_bn_mu_grad_cpu(const DTYPE *x, const DTYPE *d_x,
		const DTYPE *mean, const DTYPE *std, const DTYPE *d_rou, int num,
		int length, int channel, DTYPE *d_mean) {
	int cin_length = length / channel;

	int set;

	int m = cin_length * num;

	int n, c, f;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(n,c,f,set)
#endif
	for (n = 0; n < num; n++) {
		//iteration for channel
		for (c = 0; c < channel; c++) {
			set = n * length + cin_length * c;
			//iteration for feature map
			for (f = 0; f < cin_length; f++) {
				d_mean[c] += ((d_x[set + f] / (-std[c]))
						+ d_rou[c] * (DTYPE(-2) * (x[set + f] - mean[c]) / m));
			}
		}
	}
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
void cacu_bn_dx_grad_cpu(const DTYPE *x, const DTYPE *d_x,
		const DTYPE *mean, const DTYPE *std, const DTYPE *d_rou,
		const DTYPE *d_mean, int num, int length, int channel, DTYPE *dx) {
	int cin_length = length / channel;

	int set;

	int m = cin_length * num;

	int n, c, f;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(n,c,f,set)
#endif
	for (n = 0; n < num; n++) {
		//iteration for channel
		for (c = 0; c < channel; c++) {
			set = n * length + cin_length * c;
			//iteration for feature map
			for (f = 0; f < cin_length; f++) {

				dx[set + f] = ((d_x[set + f] / std[c])
						+ d_rou[c] * (DTYPE(2) * (x[set + f] - mean[c]) / m)
						+ (d_mean[c] / m));
			}
		}
	}
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
void cacu_bn_gamma_grad_cpu(const DTYPE *_x, const DTYPE *d_y, int num,
		int length, int channel, DTYPE *d_gamma) {
	int cin_length = length / channel;

	int set;

	int n, c, f;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(n,c,f,set)
#endif
	for (n = 0; n < num; n++) {
		//iteration for channel
		for (c = 0; c < channel; c++) {
			set = n * length + cin_length * c;
			//iteration for feature map
			for (f = 0; f < cin_length; f++) {

				d_gamma[c] += (_x[set + f] * d_y[set + f]);
			}
		}
	}
}

/**
 * @cacu_ssx
 * math y[i] *= x[i] :
 * scale by element wise.
 */
template<typename DTYPE>
void cacu_ssx_cpu(DTYPE *x, int length, DTYPE *y) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i)
		y[i] *= x[i];
}

/**
 * @cacu_group_alloc
 * alloc data by group
 */
template<typename DTYPE>
void cacu_group_alloc_cpu(int num, int channel, int channel_length,
		int group, DTYPE *y) {
	int length = channel * channel_length;

	int start_set = (channel / group) * channel_length;

	int copy_length = length - start_set;

	float_t *yp, *xp;

	for (int n = 0; n < num; ++n) {

		yp = y + n * length + start_set;
		xp = y + n * length;

		for (int i = 0; i < copy_length; ++i) {
			yp[i] = xp[i % start_set];
		}
	}
}

/**
 * @cacu_group_combine
 * combine data by group
 */
template<typename DTYPE>
void cacu_group_combine_cpu(int num, int channel, int channel_length,
		int group, DTYPE *y) {
	int length = channel * channel_length;

	int start_set = (channel / group) * channel_length;

	float_t *yp, *xp;

	for (int n = 0; n < num; ++n) {

		yp = y + n * length + start_set;
		xp = y + n * length;

		for (int i = 0; i < start_set; ++i) {
			for (int g = 0; g < group - 1; ++g)
				xp[i] += yp[i + g * start_set];
		}
	}
}

}
;


#endif
