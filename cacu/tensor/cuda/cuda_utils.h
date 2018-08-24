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

#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include "../../definition.h"

#include "cuda_log.h"

#if defined(__USE_DEVICE__) && defined(__PARALLELTYPE__)
#if  __USE_DEVICE__ == ON && __PARALLELTYPE__ == __CUDA__

using namespace std;

namespace cacu {

void cuda_set_device(unsigned int device_id);

void cuda_release();

void cuda_free(void* data_);

template<typename DTYPE>
inline DTYPE* cuda_malloc(dsize_t length) {
	DTYPE* data_;
	CUDA_CHECK(cudaMalloc((void** ) (&data_), length * sizeof(DTYPE)));
	return data_;
}

template<typename DTYPE>
inline DTYPE* cuda_malloc_v(dsize_t length, DTYPE value) {
	DTYPE* data_;
	CUDA_CHECK(cudaMalloc((void** ) (&data_), length * sizeof(DTYPE)));
	vector<DTYPE> d(length, value);
	CUDA_CHECK(
			cudaMemcpy((void* ) (data_), (void* ) (&d[0]),
					length * sizeof(DTYPE), cudaMemcpyHostToDevice));
	vector<DTYPE>().swap(d);
	return data_;
}

template<typename DTYPE>
inline void cuda_setvalue(DTYPE *data_, DTYPE value, dsize_t length) {
	vector<DTYPE> v(length, value);
	CUDA_CHECK(
			cudaMemcpy((void* ) (data_), (void* ) (&v[0]),
					length * sizeof(DTYPE), cudaMemcpyHostToDevice));
	vector<DTYPE>().swap(v);
}

template<typename DTYPE>
inline void cuda_copy2dev(DTYPE *d_data_, DTYPE* s_values, dsize_t length) {
	CUDA_CHECK(
			cudaMemcpy((void* ) (d_data_), (void* ) (s_values),
					length * sizeof(DTYPE), cudaMemcpyHostToDevice));
}

template<typename DTYPE>
inline void cuda_copy2host(DTYPE *d_data_, DTYPE* s_values, dsize_t length) {
	CUDA_CHECK(
			cudaMemcpy((void* ) (d_data_), (void* ) (s_values),
					length * sizeof(DTYPE), cudaMemcpyDeviceToHost));
}

template<typename DTYPE>
void cuda_print(DTYPE* data_, dsize_t length) {
	vector<DTYPE> v(length);
	cuda_copy2host(&v[0], data_, length);
	cout << "(";
	for (dsize_t i = 0; i < length; ++i)
		cout << v[i] << ",";
	cout << ")";
	cout << endl;
	vector<DTYPE>().swap(v);
}

template<typename DTYPE>
inline void cuda_refresh(DTYPE *data_, dsize_t length) {
	CUDA_CHECK(cudaMemset((void* ) (data_), (DTYPE ) 0, length * sizeof(DTYPE)));
}

}
#endif
#endif


#endif
