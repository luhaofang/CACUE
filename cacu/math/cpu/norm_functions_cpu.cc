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

#include "norm_functions_cpu.h"

#include <cmath>

namespace cacu {


void cacu_normalization_l1_cpu(float_t *x, const int length, const float_t epsilon) {
	float_t sum = 0;
	vec_t temp(SSIZE, 0);
	for(int i = 0; i < length; ++i)
	{
		if(i < SSIZE)
			temp[i % SSIZE] = abs(x[i]);
		else
			temp[i % SSIZE] += abs(x[i]);
	}
	int acc_length = SSIZE / 2;
	while (acc_length > 0) {
		for (int i = 0 ; i < acc_length; ++i){
			temp[i] += temp[i + acc_length];
		}
		acc_length /= 2;
	}
	sum = temp[0] + epsilon;
	for(int i = 0; i < length; ++i)
		x[i] /= sum;
}

void cacu_normalization_l1_grad_cpu(float_t *x, const int length, const float_t epsilon, float_t *yg, float_t *xg) {
	float_t sum = 0;
	vec_t temp(SSIZE, 0);
	for(int i = 0; i < length; ++i)
	{
		if(i < SSIZE)
			temp[i % SSIZE] = abs(x[i]);
		else
			temp[i % SSIZE] += abs(x[i]);
	}
	int acc_length = SSIZE / 2;
	while (acc_length > 0) {
		for (int i = 0 ; i < acc_length; ++i){
			temp[i] += temp[i + acc_length];
		}
		acc_length /= 2;
	}
	sum = temp[0] + epsilon;
	for(int i = 0; i < length; ++i)
		xg[i] = (sum - x[i]) / powf(sum, 2.0) * yg[i];
}



void cacu_normalization_l2_cpu(float_t *x, const int length, const float_t epsilon) {
	float_t sum = 0;
	vec_t temp(SSIZE, 0);
	for(int i = 0; i < length; ++i){
		if(i < SSIZE)
			temp[i % SSIZE] = powf(x[i],2.0);
		else
			temp[i % SSIZE] += powf(x[i],2.0);
	}
	int acc_length = SSIZE / 2;
	while (acc_length > 0) {
		for (int i = 0 ; i < acc_length; ++i){
			temp[i] += temp[i + acc_length];
		}
		acc_length /= 2;
	}
	sum = (sqrtf(temp[0]) + epsilon);
	for(int i = 0; i < length; ++i)
		x[i] /= sum;
}

void cacu_normalization_l2_grad_cpu(float_t *x, const int length, const float_t epsilon, float_t *yg, float_t *xg) {
	float_t sum = 0;
	vec_t temp(SSIZE, 0);
	for(int i = 0; i < length; ++i){
		if(i < SSIZE)
			temp[i % SSIZE] = powf(x[i],2.0);
		else
			temp[i % SSIZE] += powf(x[i],2.0);
	}
	int acc_length = SSIZE / 2;
	while (acc_length > 0) {
		for (int i = 0 ; i < acc_length; ++i){
			temp[i] += temp[i + acc_length];
		}
		acc_length /= 2;
	}
	sum = (sqrtf(temp[0]) + epsilon);
	float_t rsum = powf(sum, 2.0);
	float_t tsum = powf(sum, 3.0);
	for(int i = 0; i < length; ++i)
		xg[i] = (rsum - powf(x[i], 2.0)) / tsum * yg[i];
}

void cacu_norm_l1_cpu(float_t *x, const int length, const float_t epsilon, float_t *norm)
{
	vec_t temp(SSIZE, 0);
	int acc_length = SSIZE / 2;
	for(int i = 0; i < length; ++i)
	{
		if(i < SSIZE)
			temp[i % SSIZE] = abs(x[i]);
		else
			temp[i % SSIZE] += abs(x[i]);
	}
	while (acc_length > 0) {
		for (int i = 0 ; i < acc_length; ++i){
			temp[i] += temp[i + acc_length];
		}
		acc_length /= 2;
	}
	norm[0] = temp[0] + epsilon;
}

void cacu_norm_l2_cpu(float_t *x, const int length, const float_t epsilon, float_t *norm)
{
	vec_t temp(SSIZE, 0);
	int acc_length = SSIZE / 2;
	for(int i = 0; i < length; ++i){
		if(i < SSIZE)
			temp[i % SSIZE] = powf(x[i],2.0);
		else
			temp[i % SSIZE] += powf(x[i],2.0);
	}
	while (acc_length > 0) {
		for (int i = 0 ; i < acc_length; ++i){
			temp[i] += temp[i + acc_length];
		}
		acc_length /= 2;
	}
	norm[0] = (sqrtf(temp[0]) + epsilon);
}


}


