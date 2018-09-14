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

#include "../definition.h"
#include "math_functions.h"

#include "../tensor/cuda/cuda_utils.h"

namespace cacu {


/**
 * @rand_vector
 * math vector_[i] = (rand()>=ratio_)
 * length: the input data's size
 */
void rand_vector(float *vector_, const int length, const float ratio_) {

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	vec_t v_(length);

	for (int i = 0; i < length; ++i) {
		if (urand(0, 1) >= ratio_)
			v_[i] = 1;
	}
	cuda_copy2dev(vector_, &v_[0], length);
	vec_t().swap(v_);
#endif
#else
	for (int i = 0; i < length; ++i) {
		if (urand(0, 1) >= ratio_)
		vector_[i] = 1;
		else
		vector_[i] = 0;
	}
#endif

}

/**
 * @mask_vector
 * math vector_[i] = (vector_[i]>0)
 * length: the input data's size
 */
void mask_vector(float *vector_, const int length, float *mask) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	vec_t v_(length);
	cuda_copy2host(&v_[0], vector_, length);
	for (int i = 0; i < length; ++i) {
		v_[i] = 1.0 * (v_[i] > 0);
	}
	cuda_copy2dev(mask, &v_[0], length);
	vec_t().swap(v_);
#endif
#else
	for (int i = 0; i < length; ++i) {
		mask[i] = 1.0 * (vector_[i] > 0);
	}
#endif
}

/**
* @mask_vector
* math vector_[i] = (vector_[i]<0)
* length: the input data's size
*/
void mask_vector_lt(float *vector_, const int length, float *mask) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	vec_t v_(length);
	cuda_copy2host(&v_[0], vector_, length);
	for (int i = 0; i < length; ++i) {
		v_[i] = 1.0 * (v_[i] < 0);
	}
	cuda_copy2dev(mask, &v_[0], length);
	vec_t().swap(v_);
#endif
#else
	for (int i = 0; i < length; ++i) {
		mask[i] = 1.0 * (vector_[i] < 0);
	}
#endif
}

/**
 * @cacu_isaxdb
 * y[index] = x[index]*a + b
 */
void cacu_isaxb(float *x, const int channel, const int width, const int height, const float a, int *index_,
		const float b, float *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_isaxb_cuda(x, channel, width, height, a, index_, b, y);
#endif
#else
	float *xp,*yp;
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
#endif
}

unsigned int argmax(float *data, const int length) {

	unsigned int index;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	vector<float_t> _data(length);
	cuda_copy2host(&_data[0], data, length);
	float max;
	unsigned int i;
	max = _data[0];
	index = 0;
	for (i = 1; i < length; ++i) {
		if (_data[i] > max) {
			max = _data[i];
			index = i;
		}
	}
	vector<float_t>().swap(_data);
	return index;
#endif
#else
	float max;
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
#endif
}

void cacu_transpose(float *mtx, const int m, const int n, const int clength) {

//#if __USE_DEVICE__ == ON
//#if __PARALLELTYPE__ == __CUDA__
//	cacu_transpose_cuda(mtx, m, n, clength);
//#endif
//#else
	int next, cur, pre;
	vector<float> temp(clength);
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
//#endif
}


/**
 * @cacu_copy
 * math y = x:
 * length: the input data's size
 */
void cacu_copy(float *x, const int length, float *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_copy_cublas(x, length, y);
#endif
#else
	for(int i = 0;i<length; ++i)
		y[i] = x[i];
#endif
}

void cacu_clip_vec(float *data, const float threshold, const int length) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_clip_vec_cuda(data, threshold, length);
#endif
#else
	for(int i = 0; i < length; ++i)
	data[i] = data[i] * (abs(data[i]) >= threshold);
#endif
}


void cacu_abs(float *x, const int length, float *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_abs_cuda(x, length, y);
#endif
#else
	for (int i = 0; i < length; ++i)
		y[i] = abs(x[i]);
#endif
}

void cacu_abs_grad(float *x, float *diff, const int length) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_abs_grad_cuda(x, diff, length);
#endif
#else
	for (int i = 0; i < length; ++i)
		diff[i] = (x[i] > 0) ? 1 : -1;
#endif
}


}
