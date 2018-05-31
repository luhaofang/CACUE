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

#ifndef DEVICE_DATA_UTILS_H_
#define DEVICE_DATA_UTILS_H_

#include "../cuda/cuda_log.h"
#include "../cuda/cuda_utils.h"

#ifdef __USE_DEVICE__
#if  __USE_DEVICE__ == ON

namespace cacu {

inline void device_release() {
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
}

template<typename DTYPE>
inline DTYPE* device_malloc(const dsize_t length) {
#if __PARALLELTYPE__ == __CUDA__
	return cuda_malloc_v(length, 0);
#else
	return NULL;
#endif
}

template<typename DTYPE>
inline DTYPE* device_malloc_v(const dsize_t length, const DTYPE value) {
#if __PARALLELTYPE__ == __CUDA__
	return cuda_malloc_v(length, value);
#else
	return NULL;
#endif
}

template<typename DTYPE>
inline void device_setvalue(DTYPE *data_,const DTYPE value, const dsize_t length) {
#if __PARALLELTYPE__ == __CUDA__
	cuda_setvalue(data_, value, length);
#endif
}

template<typename DTYPE>
inline void device_refresh(DTYPE *data_, const dsize_t length) {
#if __PARALLELTYPE__ == __CUDA__
	cuda_refresh(data_, length);
#endif
}

template<typename DTYPE>
inline void device_copy2dev(DTYPE *d_data_, const DTYPE* s_values, const dsize_t length) {
#if __PARALLELTYPE__ == __CUDA__
	cuda_copy2dev(d_data_, s_values, length);
#endif
}

template<typename DTYPE>
inline void device_copy2host(DTYPE *d_data_,const DTYPE* s_values, const dsize_t length) {
#if __PARALLELTYPE__ == __CUDA__
	cuda_copy2host(d_data_, s_values, length);
#endif
}

template<typename DTYPE>
inline void device_free(DTYPE* data_) {
#if __PARALLELTYPE__ == __CUDA__
	cuda_free(data_);
#endif
}

template<typename DTYPE>
inline void device_print(const DTYPE* data_, const dsize_t length) {
#if __PARALLELTYPE__ == __CUDA__
	cuda_print(data_, length);
#endif
}
}

#endif
#endif


#endif
