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

#include "pooling_functions_cpu.h"

namespace cacu {

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
void cacu_max_pooling_cpu(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int output_w, const int output_h, const int channel, float_t *y,
		int* index) {
	int cout_length = output_w * output_h;
	int cin_length = input_w * input_h;
	float_t xd;
	int outset;
	int in_start, out_start;
	int i, j, c, ki, kj;
#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i,j,c,ki,kj,in_start, out_start,outset,xd,xp)
#endif
	for (i = 0; i < output_h; ++i)
		for (j = 0; j < output_w; ++j) {
			for (c = 0; c < channel; ++c) {
				out_start = (i * output_w + j);
				in_start = (i * input_w + j) * stride;
				outset = c * cout_length + out_start;
				y[outset] = x[c * cin_length + in_start];
				index[outset] = (int) (in_start);
				for (ki = 0; ki < kernel_size && (ki + i * stride) < input_h;
						++ki)
					for (kj = 0;
							kj < kernel_size && (kj + j * stride) < input_w;
							++kj) {
						xd = x[ki * input_w + kj + c * cin_length + in_start];
						if (y[outset] < xd) {
							y[outset] = xd;
							index[outset] = (int) (in_start
									+ ki * input_w + kj);
						}
					}
			}
		}
}

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
void cacu_max_pooling_grad_cpu(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int output_w, const int output_h, const int channel, float_t *y,
		const int* index) {
	int sd_out;
	int _index;

	int cout_length = output_w * output_h;
	int cin_length = input_w * input_h;

	int i, j, c;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i,j,c,_index, sd_out)
#endif
	for (i = 0; i < output_h; ++i)
		for (j = 0; j < output_w; ++j) {
			sd_out = (i * output_w + j);
			for (c = 0; c < channel; ++c) {
				_index = index[sd_out + c * cout_length];
				y[_index + c * cin_length] += x[sd_out + c * cout_length];
			}
		}
}

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
void cacu_average_pooling_cpu(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int output_w, const int output_h, const int channel, float_t *y) {
	int block_size = output_w * output_h;
	float_t *yp;
	int in_start, out_start;
	int count;

	int i, j, c, ki, kj;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i,j,c,ki,kj,in_start, out_start,count,xp,yp)
#endif
	for (c = 0; c < channel; ++c) {
		yp = y + c * block_size;
		for (i = 0; i < output_h; ++i)
			for (j = 0; j < output_w; ++j) {
				out_start = (i * output_w + j);
				in_start = (i * input_w + j) * stride;
				count = 0;
				for (ki = 0; ki < kernel_size && (ki + i * stride) < input_h;
						ki++)
					for (kj = 0;
							kj < kernel_size && (kj + j * stride) < input_w;
							kj++) {
						yp[out_start] += x[in_start + ki * input_w + kj
								+ c * input_w * input_h];
						count++;
					}
				yp[out_start] /= count;
			}
	}
}

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
void cacu_average_pooling_grad_cpu(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int output_w, const int output_h, const int channel, float_t *y) {
	int sd_out, sn_out, param_w, param_h;
	float_t *sn_out_cp;
	float_t diff_data;
	int flag_w = output_w - 1;
	int flag_h = output_h - 1;
	int pad_w = abs(input_w - (output_w - 1) * stride - kernel_size);
	int pad_h = abs(input_h - (output_h - 1) * stride - kernel_size);

	int cin_length = input_w * input_h;
	int cout_length = output_w * output_h;

	int i, j, c, ki, kj;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i,j,c,ki,kj,diff_data,sd_out_cp,sn_out_cp,param_w,param_h)
#endif
	for (i = 0; i < output_h; ++i)
		for (j = 0; j < output_w; ++j) {
			sd_out = (i * output_w + j);
			sn_out = (i * input_w + j) * stride;
			for (c = 0; c < channel; ++c) {
				//mean
				if (pad_w == 0 && pad_h == 0) {
					diff_data = x[sd_out + c * cout_length]
							/ (float_t) (kernel_size * kernel_size);
					for (ki = 0; ki < kernel_size; ++ki)
						for (kj = 0; kj < kernel_size; ++kj) {
							sn_out_cp = y + sn_out + (ki * input_w + kj)
									+ c * cin_length;
							*sn_out_cp += diff_data;
						}
				} else {
					param_w = kernel_size, param_h = kernel_size;
					if (i == flag_h)
						param_h = kernel_size - pad_h;
					if (j == flag_w)
						param_w = kernel_size - pad_w;
					diff_data = x[sd_out + c * cout_length]
							/ (float_t) (param_w * param_h);
					for (ki = 0; ki < param_w; ++ki)
						for (kj = 0; kj < param_h; ++kj) {
							sn_out_cp = y + sn_out + (ki * input_w + kj)
									+ c * cin_length;
							*sn_out_cp += diff_data;
						}
				}

			}
		}
}

void cacu_img2col_pad_cpu(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, float_t *y) {
	int cin_length = input_w * input_h;
	int kernel_length = kernel_size * kernel_size;
	int output_size = output_w * output_h;
	float_t *yp;
	int out_start;
	int i, j, c, ki, kj;
	int in_w, in_h, out_w, out_h;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i,j,c,ki,kj,yp,xp,out_start,in_start)
#endif
	for (i = 0; i < output_h; ++i)
		for (j = 0; j < output_w; ++j) {
			out_start = (i * output_w + j);
			out_h = i * stride;
			out_w = j * stride;

			for (c = 0; c < channel; ++c) {
				yp = y + c * kernel_length * output_size;
				for (ki = 0; ki < kernel_size; ++ki)
					for (kj = 0; kj < kernel_size; ++kj) {
						in_h = out_h + ki;
						in_w = out_w + kj;
						if (in_w >= pad_w && in_w < input_w + pad_w
								&& in_h >= pad_h && in_h < input_h + pad_h)
							yp[(ki * kernel_size + kj) * output_size + out_start] =
									x[(in_h - pad_h) * input_w + in_w - pad_w
											+ c * cin_length];
					}
			}
		}
}

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *stride: stride move of the kernel
 *input_dim: width of input data
 *output_dim: width of output data
 */
void cacu_col2img_pad_cpu(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, float_t *y) {
	int sd_out, sn_out;

	int block_size = kernel_size * kernel_size * channel;
	int k_size = kernel_size * kernel_size;
	int cout_length = output_w * output_h;
	int cin_length = input_w * input_h;
	float_t *yp;

	int row, col, c, ki, kj;
	int in_h, in_w, out_h, out_w;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(row,col,c,ki,kj,yp,xp,sd_out,sn_out)
#endif
	//for output_dim's location
	for (row = 0; row < output_h; ++row) {
		for (col = 0; col < output_w; ++col) {

			out_h = row * stride;
			out_w = col * stride;
			sd_out = (row * output_w + col);
			for (c = 0; c < channel; ++c) {
				yp = y + c * cin_length;
				for (ki = 0; ki < kernel_size; ++ki)
					for (kj = 0; kj < kernel_size; ++kj) {
						in_h = out_h + ki;
						in_w = out_w + kj;
						if (in_w >= pad_w && in_w < input_w + pad_w
								&& in_h >= pad_h && in_h < input_h + pad_h)
							yp[(in_h - pad_h) * input_w + in_w - pad_w] += x[(ki
									* kernel_size + kj + c * k_size)
									* cout_length + sd_out];
					}
			}
		}
	}
}

void cacu_row_max_pooling_cpu(float_t *x, const int input_length,
		const int output_length, float_t *y) {
	sort(x, x + input_length);
	for (int i = 0; i < output_length; ++i) {
		y[i] = x[i];
	}
}

void cacu_row_max_pooling_index_cpu(float_t *x, const int input_length,
		const int output_length, float_t *y, int* index) {
	for (int i = 0; i < output_length; ++i) {
		for (int j = 0; j < input_length; ++j)
			if (x[j] == y[i]) {
				index[i] = j;
				break;
			}
	}
}

void cacu_row_max_pooling_grad_cpu(const float_t *x, const int output_length,
		float_t *y, const int* index) {
	for (int i = 0; i < output_length; ++i) {
		y[index[i]] = x[i];
	}
}

}
