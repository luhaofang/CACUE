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

/**
 * for activation use relu functions in cuda
 */
extern "C" void cacu_relu_gpu(mycnn::float_t *x, int length);

/**
 * gradient for activation use relu functions in cuda
 */
extern "C" void cacu_relu_grad_gpu(mycnn::float_t *x, mycnn::float_t *g, int length);

/**
 * for activation use leaky_relu functions in cuda
 */
extern "C" void cacu_leaky_relu_gpu(mycnn::float_t *x, mycnn::float_t a, int length);

/**
 * gradient for activation use leaky_relu functions in cuda
 */
extern "C" void cacu_leaky_relu_grad_gpu(mycnn::float_t *x, mycnn::float_t *g, mycnn::float_t a, int length);

/**
 * for activation use softmax functions in cuda
 */
extern "C" void cacu_softmax_gpu(mycnn::float_t *x, int num , int length, mycnn::float_t *y);

/**
 * for activation use tanh functions in cuda
 */
extern "C" void cacu_tanh_gpu(mycnn::float_t *x, int length, mycnn::float_t *y);

/**
 * gradient for activation use tanh functions in cuda
 */
extern "C" void cacu_tanh_grad_gpu(mycnn::float_t *x, mycnn::float_t *g, int length, mycnn::float_t *y);

/**
 * for activation use sigmoid functions in cuda
 */
extern "C" void cacu_sigmoid_gpu(mycnn::float_t *x, int length, mycnn::float_t *y);

/**
 * gradient for activation use sigmoid functions in cuda
 */
extern "C" void cacu_sigmoid_grad_gpu(mycnn::float_t *x, mycnn::float_t *g, int length, mycnn::float_t *y);

