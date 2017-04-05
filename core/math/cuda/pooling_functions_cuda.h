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


/**
 * @cacu_max_pooling_gpu
 * channel: channel of input data
 * kernel_size: pooling window size
 * input_dim: width of input data
 * output_dim: width of output data
 * index: the array of the maximum set
 */
extern "C" void cacu_max_pooling_gpu(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y, unsigned int* index);

/**
 * @cacu_max_pooling_gpu
 * channel: channel of input data
 * kernel_size: pooling window size
 * input_dim: width of input data
 * output_dim: width of output data
 * index: the array of the maximum set
 */
extern "C" void cacu_max_pooling_grad_gpu(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y, unsigned int* index);


/**
 * @cacu_average_pooling_gpu
 * channel: channel of input data
 * kernel_size: pooling window size
 * input_dim: width of input data
 * output_dim: width of output data
 */
extern "C" void cacu_average_pooling_gpu(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y);

/**
 * @cacu_average_pooling_gpu
 * channel: channel of input data
 * kernel_size: pooling window size
 * input_dim: width of input data
 * output_dim: width of output data
 */
extern "C" void cacu_average_pooling_grad_gpu(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y);

/**
 * @cacu_padded_data_gpu
 * pad data.
 * y is input_dim + 2 * pad dim wise
 */
extern "C" void cacu_padded_data_gpu(float_t *x,int channel, int input_dim, int pad, float_t *y);

/**
 * @cacu_img2col_gpu
 * img2col
 */
extern "C" void cacu_img2col_gpu(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim, float_t *y);


extern "C" void cacu_img2col_pad_gpu(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim,int pad,float_t *y);
/**
 * @cacu_unpadded_data_gpu
 * unpadded data.
 * y is input_dim - 2 * pad dim wise
 */
extern "C" void cacu_unpadded_data_gpu(float_t *x,int channel, int input_dim, int pad, float_t *y);

/**
 * @cacu_img2col_gpu
 * col2img
 */
extern "C" void cacu_col2img_gpu(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim, float_t *y);

extern "C" void cacu_col2img_pad_gpu(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim,int pad, float_t *y);

