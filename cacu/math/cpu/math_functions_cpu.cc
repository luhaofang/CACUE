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

#include "math_functions_cpu.h"

#include "../../config.h"
#include "../../definition.h"
#include "math_functions_cpu.h"
#include "../utils/rand_t.h"

#include "../../tensor/cuda/cuda_utils.h"
#include "../../tensor/utils/device_data_utils.h"
#include "../../tensor/utils/cpu_data_utils.h"
#include "../cuda/math_functions_cuda.h"
#include "../cuda/activation_functions_cuda.h"

namespace cacu {


/**
 * @rand_vector
 * math vector_[i] = (rand()>=ratio_)
 * length: the input data's size
 */
void rand_vector_cpu(float_t *vector_, const int length, const float_t ratio_) {

	for (int i = 0; i < length; ++i) {
		if (urand(0, 1) >= ratio_)
		vector_[i] = 1;
		else
		vector_[i] = 0;
	}
}

/**
 * @mask_vector
 * math vector_[i] = (vector_[i]>0)
 * length: the input data's size
 */
void mask_vector_cpu(float_t *vector_, const int length, float_t *mask) {
	for (int i = 0; i < length; ++i) {
		mask[i] = 1.0 * (vector_[i] > 0);
	}
}

/**
* @mask_vector
* math vector_[i] = (vector_[i]<0)
* length: the input data's size
*/
void mask_vector_lt_cpu(float_t *vector_, const int length, float_t *mask) {

	for (int i = 0; i < length; ++i) {
		mask[i] = 1.0 * (vector_[i] < 0);
	}
}

/**
 * @cacu_isaxdb
 * y[index] = x[index]*a + b
 */
void cacu_isaxb_cpu(float_t *x, const int channel, const int width, const int height, const float_t a, int *index_,
		const float_t b, float_t *y) {
	float_t *xp,*yp;
	int c_length = width * height;
	int length = channel* c_length;
	int index;
	if(*index_ >= 0 ) {
		for(int i = 0 ; i< length; ++i)
			y[i] = x[i];
		for (int h = 0 ; h < height; ++h)
			for(int w =0; w < width; ++w)
			{
				index =  h * width + w;
				xp = x + index;
				yp = y + index;
				yp[index_[index]*c_length] = a * xp[index_[index]*c_length] + b;
			}
	}
	else
	{
		for (int i = 0; i < length; ++i)
			y[i] = 0;
	}
}

unsigned int argmax_cpu(float_t *data, const int length) {

	unsigned int index;
	float_t max;
	unsigned int i;
	max = data[0];
	index = 0;
	for (i = 1; i < length; ++i) {
		if (data[i] > max) {
			max = data[i];
			index = i;
		}
	}
	return index;
}

void cacu_transpose_cpu(float_t *mtx, const int m, const int n, const int clength) {

	int next, cur, pre;
	vector<float_t> temp(clength);
	for (int i = 0; i < m * n; ++i) {
		next = (i % n) * m + i / n;
		while (next > i)
		next = (next % n) * m + next / n;
		if (next == i) {
			for(int n = 0 ; n < clength ; ++n)
				temp[n] = mtx[i*clength + n];
			cur = i;
			pre = (cur % m) * n + cur / m;
			while (pre != i) {
				for(int n = 0 ; n < clength ; ++n)
					mtx[cur*clength + n] = mtx[pre*clength + n];
				cur = pre;
				pre = (cur % m) * n + cur / m;
			}
			for(int n = 0 ; n < clength ; ++n)
				mtx[cur*clength +n] = temp[n];
		}
	}
}

void cacu_rotate_cpu(float_t *mtx, const int m, const int n, const int clength, rotate_code rotate){

	vector<float_t> temp(clength);
	vector<float_t> temp1(clength);
	int curpos, despos;
	int w = m / 2;
	switch(rotate) {
		case rotate_top:
//			cout << "top" << endl;
			break;
		case rotate_left:
			for (int i = 0; i < w; ++i)
				for (int j = i; j < n - i - 1; ++j) {
					curpos = i * n + i + j;
					despos = (m - j - i - 1) * n + i;
					for(int c = 0 ; c < clength; ++c){
						temp[c] = mtx[m * n * c + despos];
						mtx[m * n * c + despos] = mtx[m * n * c + curpos];
					}
					curpos = despos;
					despos = (m - i - 1) * n + (n - i - j - 1);
					for(int c = 0 ; c < clength; ++c){
						temp1[c] = mtx[m * n * c + despos];
						mtx[m * n * c + despos] = temp[c];
					}
					curpos = despos;
					despos = (j + i) * n + (n - i - 1);
					for(int c = 0 ; c < clength; ++c){
						temp[c] = mtx[m * n * c + despos];
						mtx[m * n * c + despos] = temp1[c];
					}
					curpos = despos;
					despos = i * n + i + j;
					for(int c = 0 ; c < clength; ++c){
						temp1[c] = mtx[m * n * c + despos];
						mtx[m * n * c + despos] = temp[c];
					}
				}
//			cout << "left" << endl;
			break;
		case rotate_bottom:
			for (int i = 0; i < w; ++i)
				for (int j = i; j < n - i - 1; ++j) {
					curpos = i * n + i + j;
					despos = (m - i - 1) * n + (n - i - j - 1);
					for(int c = 0 ; c < clength; ++c){
						temp[c] = mtx[m * n * c + despos];
						mtx[m * n * c + despos] = mtx[m * n * c + curpos];
						mtx[m * n * c + curpos] = temp[c];
					}

					curpos = (m - j - i - 1) * n + i;
					despos = (j + i) * n + (n - i - 1);
					for(int c = 0 ; c < clength; ++c){
						temp[c] = mtx[m * n * c + despos];
						mtx[m * n * c + despos] = mtx[m * n * c + curpos];
						mtx[m * n * c + curpos] = temp[c];
					}
				}
//			cout << "bottom" << endl;
			break;
		case rotate_right:
			for (int i = 0; i < w; ++i)
				for (int j = i; j < n - i - 1; ++j) {
					curpos = i * n + i + j;
					despos = (j + i) * n + (n - i - 1);
					for(int c = 0 ; c < clength; ++c){
						temp[c] = mtx[m * n * c + despos];
						mtx[m * n * c + despos] = mtx[m * n * c + curpos];
					}
					curpos = despos;
					despos = (m - i - 1) * n + (n - i - j - 1);
					for(int c = 0 ; c < clength; ++c){
						temp1[c] = mtx[m * n * c + despos];
						mtx[m * n * c + despos] = temp[c];
					}
					curpos = despos;
					despos = (m - j - i - 1) * n + i;
					for(int c = 0 ; c < clength; ++c){
						temp[c] = mtx[m * n * c + despos];
						mtx[m * n * c + despos] = temp1[c];
					}
					curpos = despos;
					despos = i * n + i + j;
					for(int c = 0 ; c < clength; ++c){
						temp1[c] = mtx[m * n * c + despos];
						mtx[m * n * c + despos] = temp[c];
					}
				}
//			cout << "right" << endl;
			break;
		default:
			break;
	}
}


/**
 * @cacu_copy
 * math y = x:
 * length: the input data's size
 */
void cacu_copy_cpu(float_t *x, const int length, float_t *y) {

	for(int i = 0;i<length; ++i)
		y[i] = x[i];
}

/**
 * @cacu_clip_vec
 * math vector_[i] belongs to [lt, rt]
 * length: the input data's size
 */
void cacu_clip_vec_cpu(float_t *data, const float_t l_t, const float_t r_t, const int length) {
	assert(l_t < r_t);
	for(int i = 0; i < length; ++i) {
		data[i] = data[i] >= l_t ? data[i] : l_t;
		data[i] = data[i] <= r_t ? data[i] : r_t;
	}
}


void cacu_abs_cpu(float_t *x, const int length, float_t *y) {

	for (int i = 0; i < length; ++i)
		y[i] = abs(x[i]);
}

void cacu_abs_grad_cpu(float_t *x, float_t *diff, const int length) {

	for (int i = 0; i < length; ++i)
		diff[i] = (x[i] >= 0) ? 1 : -1;
}

void cacu_pow_cpu(float_t *x, const int length, const float_t pow, float_t *y){

	for (int i = 0; i < length; ++i)
		y[i] = powf(x[i], pow);
}

void cacu_powt_cpu(float_t *x, const int length, const float_t *pow, float_t *y){
	for (int i = 0; i < length; ++i)
		y[i] = powf(x[i], pow[i]);
}

void cacu_ln_cpu(float_t *x, const int length, float_t *y){
	for (int i = 0; i < length; ++i)
		y[i] = logf(x[i]);
}

void cacu_flip_cpu(int *x, const int length){
	for (int i = 0; i < length; ++i)
		x[i] = (x[i] == 0);
}

void cacu_cos_cpu(float_t *x, const int length, float_t *y) {
	for (int i = 0; i < length; ++i)
		y[i] = cosf(x[i]);
}

void cacu_acos_cpu(float_t *x, const int length, float_t *y) {
	for (int i = 0; i < length; ++i)
		y[i] = acosf(x[i]);
}

void cacu_floor_cpu(float_t *x, const int length, float_t *	y){
	for (int i = 0; i < length; ++i)
		y[i] = floorf(x[i]);
}


}


