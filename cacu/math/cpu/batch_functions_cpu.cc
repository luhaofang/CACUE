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

#include "batch_functions_cpu.h"

#include <math.h>

namespace cacu {


/**
 * @cacu_sumbysize
 * sum by size:
 * accumulate the value by width or height , width is the matrix array's width dim which stored in row -major format.
 * sum by width y is (length/ width) height dim, sum by height y is width dim.
 * warning: take seriously this function may create accumulated error when width is large enough
 */

void cacu_sumbysize_cpu(SUM SUMTYPE, float_t *x, int length,
		const float_t alpha, float_t *y, const float_t beta, int width) {
	int height = length / width;
	int b, i;
	float_t acc;
	if (BYWIDTH == SUMTYPE) {
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(b,i,acc)
#endif
		for (b = 0; b < height; ++b) {
			acc = float_t(0);
			for (i = 0; i < width; ++i)
				acc += x[i + b * width];
			y[b] = ((alpha * acc) + beta * y[b]);
		}
	} else if (BYHEIGHT == SUMTYPE) {
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(b,i,acc)
#endif
		for (b = 0; b < width; ++b) {
			acc = float_t(0);
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

void cacu_cxsize_cpu(float_t *x, int length, float_t *a,
		int size, float_t *y) {
	int block_size = length / size;
	int b;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(b)
#endif
	if(length == 1 && size == 1)
		y[0] = x[0] * a[0];
	else{
		for (b = 0; b < length; ++b) {
			y[b] = x[b] * a[b / block_size];
		}
	}
}

/**
 * @cacu_cdxsize
 * math y[i] = x[i] / a[j] :
 * x is a length dim array list, a is a size dim array list, a[j] is the corresponding denominator, j = i / (length / size).
 */

void cacu_cdxsize_cpu(float_t *x, int length, float_t *a,
		int size, float_t *y) {
	int block_size = length / size;
	int b;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(b)
#endif
	if(length == 1 && size == 1)
		y[0] = x[0] / a[0];
	else{
		for (b = 0; b < length; ++b) {
			y[b] = x[b] / a[b / block_size];
		}
	}
}

/**
 * @cacu_sdxsize
 * math y[i] = (x[i] + a) / b:
 * x is a length dim array list, a is the corresponding denominator.
 */

void cacu_sdxsize_cpu(float_t *x, int length, float_t a, float_t b, float_t *y) {
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

void cacu_ssxpy_cpu(float_t *x, const float_t a, int size,
		float_t *y, const float_t b, int length, float_t *z) {
	int block_size = length / size;

	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif

	if(length == 1 && size == 1)
		z[0] = b * y[0] + a * x[0];
	else{
		for (i = 0; i < length; ++i) {
			z[i] = b * y[i] + a * x[i / block_size];
		}
	}
}

/**
 * @cacu_sqr
 * math y[i] = x[i]^2 :
 */

void cacu_sqr_cpu(float_t *x, int length, float_t *y) {
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

void cacu_root_cpu(float_t *x, int length, float_t *y) {

	int j;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(j)
#endif
	for (j = 0; j < length; ++j)
		y[j] = sqrtf((float_t) x[j]);
}

/**
 * @cacu_stdbychannel
 * math std[i] = sqrt(varience[i] + epsilon) :
 */

void cacu_stdbychannel_cpu(float_t *varience, int length, float_t *std,
		const float_t epsilon) {
	int j;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(j)
#endif
	for (j = 0; j < length; ++j)
		std[j] = (float_t) sqrtf(varience[j] + epsilon);
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

void cacu_bn_rou_grad_cpu(const float_t *x, const float_t *d_x,
		const float_t *mean, const float_t *std, int num, int length, int channel,
		float_t *d_rou) {
	int cin_length = length / channel;

	int m = num * cin_length;

	int set;

	int data_row, data_col;

	int c, f;

	vec_t temp(SSIZE, 0);
	int acc_length;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(c,f,set)
#endif
	//iteration for channel
	for (c = 0; c < channel; c++) {
		//iteration for feature map
		for (f = 0; f < m; f++) {
			data_row = f / cin_length;
			data_col = f % cin_length;
			set = data_row * length + data_col + c * cin_length;
			if(f < SSIZE)
				temp[f % SSIZE] = ((x[set] - mean[c]) * d_x[set]);
			else
				temp[f % SSIZE] += ((x[set] - mean[c]) * d_x[set]);
			//d_rou[c] += (mean[c] - x[set]) * d_x[set];
		}
		acc_length = SSIZE / 2;
		while (acc_length > 0) {
			for (int i = 0 ; i < acc_length; ++i){
				temp[i] += temp[i + acc_length];
			}
			acc_length /= 2;
		}
		d_rou[c] += (temp[0] * (float_t(-0.5) / (std[c] * std[c] * std[c])));
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

void cacu_bn_mu_grad_cpu(const float_t *x, const float_t *d_x,
		const float_t *mean, const float_t *std, const float_t *d_rou, int num,
		int length, int channel, float_t *d_mean) {
	int cin_length = length / channel;

	int set;

	int m = cin_length * num;

	int c, f;
	int data_row, data_col;

	vec_t temp(SSIZE, 0);
	int acc_length;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(c,f,set)
#endif

	//iteration for channel
	for (c = 0; c < channel; c++) {
		//iteration for feature map
		for (f = 0; f < m; f++) {
			data_row = f / cin_length;
			data_col = f % cin_length;
			set = data_row * length + data_col + c * cin_length;
			if(f < SSIZE)
				temp[f % SSIZE] = ((d_x[set] / (-std[c]))
						+ (d_rou[c] / m) * ((float_t(-2.0)) * (x[set] - mean[c])));
			else
				temp[f % SSIZE] += ((d_x[set] / (-std[c]))
						+ (d_rou[c] / m) * ((float_t(-2.0)) * (x[set] - mean[c])));
			//d_mean[c] += ((d_x[set] / (-std[c])) + (d_rou[c] / m) * ((-2.0) * (x[set] - mean[c])));
		}
		acc_length = SSIZE / 2;
		while (acc_length > 0) {
			for (int i = 0 ; i < acc_length; ++i)
				temp[i] += temp[i + acc_length];
			acc_length /= 2;
		}
		d_mean[c] += temp[0];
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

void cacu_bn_dx_grad_cpu(const float_t *x, const float_t *d_x,
		const float_t *mean, const float_t *std, const float_t *d_rou,
		const float_t *d_mean, int num, int length, int channel, float_t *dx) {
	int cin_length = length / channel;

	int m = cin_length * num;

	int c, i;

	int count = num * length;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i,c)
#endif
	for (i = 0; i < count; ++i) {
		c = (i % length) / cin_length;
		dx[i] = ((d_x[i] / std[c])
				+ d_rou[c] * (float_t(2.0) * (x[i] - mean[c]) / m)
				+ (d_mean[c] / m));
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

void cacu_bn_gamma_grad_cpu(const float_t *_x, const float_t *d_y, int num,
		int length, int channel, float_t *d_gamma) {
	int cin_length = length / channel;

	int set;

	int c, f;
	int m = num * cin_length;
	int data_row, data_col;

	vec_t temp(SSIZE, 0);
	int acc_length;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(c,f,set)
#endif

	for (c = 0; c < channel; c++) {
		//iteration for feature map
		for (f = 0; f < m; f++) {
			data_row = f / cin_length;
			data_col = f % cin_length;
			set = data_row * length + data_col + c * cin_length;
			if(f < SSIZE)
				temp[f % SSIZE] = (_x[set] * d_y[set]);
			else
				temp[f % SSIZE] += (_x[set] * d_y[set]);
			//d_gamma[c] += (_x[set] * d_y[set]);
		}
		acc_length = SSIZE / 2;
		while (acc_length > 0) {
			for (int i = 0 ; i < acc_length; ++i){
				temp[i] += temp[i + acc_length];
			}
			acc_length /= 2;
		}
		d_gamma[c] += temp[0];
	}
}

/**
 * @cacu_ssx
 * math y[i] *= x[i] :
 * scale by element wise.
 */

void cacu_ssx_cpu(float_t *x, int length, float_t *y) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i)
		y[i] *= x[i];
}

/**
 * @cacu_invx
 * math y[i] = 1 / x[i] :
 * scale by element wise.
 */

void cacu_invx_cpu(float_t *x, int length, float_t *y) {
	int i;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i)
#endif
	for (i = 0; i < length; ++i)
		y[i] = 1.0 / x[i];
}

/**
 * @cacu_group_alloc
 * alloc data by group
 */

void cacu_group_alloc_cpu(int num, int channel, int channel_length,
		int group, float_t *y) {
	int length = channel * channel_length;

	int start_set = (channel / group) * channel_length;

	int copy_length = length - start_set;

	float_t *yp, *xp;

	int n;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(n, xp, yp)
#endif
	for (n = 0; n < num; ++n) {

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

void cacu_group_combine_cpu(int num, int channel, int channel_length,
		int group, float_t *y) {
	int length = channel * channel_length;

	int start_set = (channel / group) * channel_length;

	int n;

	float_t *yp, *xp;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(n, xp, yp)
#endif
	for (n = 0; n < num; ++n) {

		yp = y + n * length + start_set;
		xp = y + n * length;

		for (int i = 0; i < start_set; ++i) {
			for (int g = 0; g < group - 1; ++g)
				xp[i] += yp[i + g * start_set];
		}
	}
}

}

