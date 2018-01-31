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

#include "../../config.h"

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__

#include "../../definition.h"
#include "../math_definition.h"

namespace cacu{

/**
 * @cacu_sumbysize_cuda
 * sum by size:
 * accumulate the value by width or height , width is the matrix array's width dim which stored in row -major format.
 * sum by width y is (length/ width) height dim, sum by height y is width dim.
 */
extern "C" void cacu_sumbysize_cuda(SUM SUMTYPE ,const float_t *x, int length,const float_t alpha, float_t *y,const float_t beta, int width);

/**
 * @cacu_cxsize_cuda
 * math y[i] = a[j]*x[i] :
 * x is a length dim array list, a is a size dim array list, a[j] is the corresponding scalar, j = i / (length / size).
 */
extern "C" void cacu_cxsize_cuda(const float_t *x, int length,const float_t *a, int size,float_t *y);

/**
 * @cacu_sxsize_cuda
 * math y[i] = a*x[i] :
 * x is a length dim array list, a is the corresponding scalar.
 */
extern "C" void cacu_sxsize_cuda(const float_t *x, int length,const float_t a, float_t *y);

/**
 * @cacu_cdxsize_cuda
 * math y[i] = x[i] / a[j] :
 * x is a length dim array list, a is a size dim array list, a[j] is the corresponding denominator, j = i / (length / size).
 */
extern "C" void cacu_cdxsize_cuda(const float_t *x, int length,const float_t *a, int size, float_t *y);

/**
 * @cacu_ssxpy_cuda
 * math z[i] = a * x[j] + b * y[i] :
 * y is a length dim array list, x is a size dim array list, x[j] is the corresponding scalar, j = i / (length / size).
 * a & b are corresponding scalars for x, y
 */
extern "C" void cacu_ssxpy_cuda(const float_t *x,const float_t a, int size,const float_t *y,const float_t b, int length, float_t *z);

/**
 * @cacu_sqr_cuda
 * math y[i] = x[i]^2 :
 */
extern "C" void cacu_sqr_cuda(const float_t *x, int length, float_t *y);

/**
 * @cacu_root_cuda
 * math y[i] = sqrt(x[i]) :
 */
extern "C" void cacu_root_cuda(const float_t *x, int length, float_t *y);

/**
 * @cacu_stdbychannel_cuda
 * math std[i] = sqrt(varience[i] + epsilon) :
 */
extern "C" void cacu_stdbychannel_cuda(const float_t *varience, int length, float_t *std,const float_t epsilon);

/**
 * @cacu_bn_rou_grad_cuda
 * calculate the gradient of bn layer's rou
 * x: input feature
 * d_x: gradient of ^x
 * mean: mean of batch
 * std: standard deviation of batch
 * length: size of a feature map
 * d_rou: gradient of batch's variance
 */
extern "C" void cacu_bn_rou_grad_cuda(const float_t *x,const float_t *d_x,const float_t *mean,const float_t *std, int num, int length, int channel, float_t *d_rou);

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
extern "C" void cacu_bn_mu_grad_cuda(const float_t *x,const float_t *d_x,const float_t *mean,const float_t *std,const float_t *d_rou, int num, int length, int channel,float_t *d_mean);

/**
 * @cacu_bn_dx_grad_cuda
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
extern "C" void cacu_bn_dx_grad_cuda(const float_t *x,const float_t *d_x,const float_t *mean,const float_t *std,const float_t *d_rou,const float_t *d_mean, int num, int length, int channel,float_t *dx);

/**
 * @cacu_bn_gamma_grad_cuda
 * calculate the gradient of bn layer's scale
 * _x: is ^x
 * d_y: gradient propagate form top layer
 * length: size of a feature map
 * d_gamma: gradient of gamma
 */
extern "C" void cacu_bn_gamma_grad_cuda(const float_t *_x,const float_t *d_y, int num, int length, int channel, float_t *d_gamma);

/**
 * @cacu_ssx_cuda
 * math y[i] *= x[i] :
 * scale by element wise.
 */
extern "C" void cacu_ssx_cuda(const float_t *x, int length, float_t *y);

/**
 * @cacu_group_alloc
 * alloc data by group
 */
extern "C" void cacu_group_alloc_cuda(int num, int channel, int channel_length, int group, float_t *y);



/**
 * @cacu_group_combine
 * combine data by group
 */
extern "C" void cacu_group_combine_cuda(int num, int channel, int channel_length, int group, float_t *y);


}


#endif
#endif


