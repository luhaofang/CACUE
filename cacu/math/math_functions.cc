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

#include "math_functions.hpp"

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
		if (v_[i] > 0)
			v_[i] = 1.0;
	}
	cuda_copy2dev(mask, &v_[0], length);
	vec_t().swap(v_);
#endif
#else
	for (int i = 0; i < length; ++i) {
		if (vector_[i] > 0)
		mask[i] = 1.0;
	}
#endif
}

/**
 * @cacu_isaxdb
 * y[index] = x[index]*a + b
 */
void cacu_isaxb(float *x, const int length, const float a,
		unsigned int *index_, const float b, float *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_isaxb_cuda(x, length, a, index_, b, y);
#endif
#else
	for (int i = 0; i < length; ++i)
	y[i] = x[i];
	y[*index_] = a * x[*index_] + b;
#endif
}

unsigned int argmax(float *data, const int length) {

	unsigned int index;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	unsigned int *_index = cuda_malloc<unsigned int>(1);
	cacu_argmax_cuda(data, length, _index);
	cuda_copy2host(&index, _index, 1);
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

void cacu_transpose(float *mtx, const int m, const int n) {

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_transpose_cuda(mtx, m, n);
#endif
#else
	int next, cur, pre;
	float temp;
	for (int i = 0; i < m * n; ++i) {
		next = (i % n) * m + i / n;
		while (next > i)
		next = (next % n) * m + next / n;
		if (next == i) {
			temp = mtx[i];
			cur = i;
			pre = (cur % m) * n + cur / m;
			while (pre != i) {
				mtx[cur] = mtx[pre];
				cur = pre;
				pre = (cur % m) * n + cur / m;
			}
			mtx[cur] = temp;
		}
	}
#endif
}

/**
 * @cacu_copy
 * math y = x:
 * length: the input data's size
 */
void cacu_copy(float *x, const int length, float *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_copy_cublas(x,length,y);
#endif
#else
	memcpy(y, x, length * sizeof(float));
#endif
}

void cacu_clip_vec(float *data, const float threshold,
		const int length) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_clip_vec_cuda(data, threshold, length);
#endif
#else
	for(int i = 0; i < length; ++i)
	data[i] = data[i] * (abs(data[i]) >= threshold);
#endif
}

}
