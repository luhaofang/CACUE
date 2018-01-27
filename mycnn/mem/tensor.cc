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

#include "tensor.h"
#include "utils/cpu_data_utils.hpp"
#include "utils/device_data_utils.hpp"


namespace mycnn {


template<class DTYPE>
tensor<DTYPE>::tensor(size_t length){

	_length = length;
#if __USE_DEVICE__ == ON
	_pdata = device_malloc_v<DTYPE>(length, 0);
#else
	_pdata = (DTYPE*)malloc(length * sizeof(DTYPE));
	set_data(0);
#endif

}

template<class DTYPE>
tensor<DTYPE>::~tensor(){

#if __USE_DEVICE__ == ON
	device_free(_pdata);
#else
	free(_pdata);
#endif

}

template<class DTYPE>
inline void tensor<DTYPE>::resize(size_t length){
#if __USE_DEVICE__ == ON
	device_free(_pdata);
	_pdata = device_malloc_v<DTYPE>(length, 0);
#else
	free(_pdata);
	_pdata = (DTYPE*)malloc(_length * sizeof(DTYPE));
	set_data(0);
#endif
}

template<class DTYPE>
inline void tensor<DTYPE>::copy2data(DTYPE* data_){
#if __USE_DEVICE__ == ON
	device_copy2dev(_pdata, &data_[0], _length);
#else
	cacu_copy(&data_[0],_length, _pdata);
#endif
}

template<class DTYPE>
inline void tensor<DTYPE>::copy2data(size_t sp, size_t length, DTYPE* data_){
#if __USE_DEVICE__ == ON
	device_copy2dev(_pdata + sp, &data_[0], length);
#else
	cacu_copy(&data_[0],length, _pdata + sp);
#endif
}


template<class DTYPE>
inline void tensor<DTYPE>::set_value(DTYPE value)
{
#if __USE_DEVICE__ == ON
	device_setvalue<DTYPE>(_pdata, value, _length);
#else
	for(size_t i = 0; i < _length; ++i)
		_pdata[i] = value;
#endif
}

template<class DTYPE>
inline void tensor<DTYPE>::set_value(size_t sp, size_t length, DTYPE value)
{
#if __USE_DEVICE__ == ON
	device_setvalue<DTYPE>(_pdata + sp, value, length);
#else
	for(size_t i = sp; i < length; ++i)
		_pdata[i] = value;
#endif
}

template<class DTYPE>
inline void tensor<DTYPE>::refresh()
{
#if __USE_DEVICE__ == ON
	device_refresh(_pdata, _length);
#else
	cacu_memset<DTYPE>(_pdata, 0 ,_length);
#endif
}


}
