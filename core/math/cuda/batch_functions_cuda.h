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
 * @cacu_sumbysize_gpu
 * sum by size:
 * accumulate the value by width or height , width is the matrix array's width dim which stored in row -major format.
 * sum by width y is (length/ width) height dim, sum by height y is width dim.
 */
extern "C" void cacu_sumbysize_gpu(SUM SUMTYPE ,float_t *x, int length, float_t *y, int width);

/**
 * @cacu_cxsize_gpu
 * math y[i] = a[j]*x[i] :
 * x is a length dim array list, a is a size dim array list, a[j] is the corresponding scalar, j = i / (length / size).
 */
extern "C" void cacu_cxsize_gpu(float_t *x, int length, float_t *a, int size,float_t *y);

/**
 * @cacu_sxsize_gpu
 * math y[i] = a*x[i] :
 * x is a length dim array list, a is the corresponding scalar.
 */
extern "C" void cacu_sxsize_gpu(float_t *x, int length, float_t a, float_t *y);

/**
 * @cacu_cdxsize_gpu
 * math y[i] = x[i] / a[j] :
 * x is a length dim array list, a is a size dim array list, a[j] is the corresponding denominator, j = i / (length / size).
 */
extern "C" void cacu_cdxsize_gpu(float_t *x, int length, float_t *a, int size, float_t *y);

/**
 * @cacu_sdxsize_gpu
 * math y[i] = x[i] / a :
 * x is a length dim array list, a is the corresponding denominator.
 */
extern "C" void cacu_sdxsize_gpu(float_t *x, int length, float_t a, float_t *y);

/**
 * @cacu_ssxpy_gpu
 * math z[i] = a * x[j] + b * y[i] :
 * y is a length dim array list, x is a size dim array list, x[j] is the corresponding scalar, j = i / (length / size).
 * a & b are corresponding scalars for x, y
 */
extern "C" void cacu_ssxpy_gpu(float_t *x, float_t a, int size, float_t *y, float_t b, int length, float_t *z);

/**
 * @cacu_sqr_gpu
 * math y[i] = x[i]^2 :
 */
extern "C" void cacu_sqr_gpu(float_t *x, int length, float_t *y);

/**
 * @cacu_root_gpu
 * math y[i] = sqrt(x[i]) :
 */
extern "C" void cacu_root_gpu(float_t *x, int length, float_t *y);

/**
 * @cacu_stdbychannel_gpu
 * math std[i] = sqrt(varience[i] + epsilon) :
 */
extern "C" void cacu_stdbychannel_gpu(float_t *varience, int length, float_t *std, float_t epsilon);

/**
 * @cacu_ssx_gpu
 * math y[i] *= x[i] :
 * scale by element wise.
 */
extern "C" void cacu_ssx_gpu(float_t *x, int length, float_t *y);
