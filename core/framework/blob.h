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

#include <vector>

#include "../math/cuda/cuda_utils.h"


using namespace std;

namespace mycnn{

	class blob : public blob_base{

	public:

		blob(int num, int channel, int width, int height, float_t _value=0, phrase_type phrase=test)
			:blob_base(num, channel, width, height, phrase){
#if __PARALLELTYPE__ == __GPU__
			_s_data = cuda_malloc_v<float_t>(num*channel*width*height, _value);
			CUDA_CHECK(res);
			if (train == phrase){
				_s_diff = cuda_malloc<float_t>(num*channel*width*height);
				CUDA_CHECK(res);
			}
#else
			_data.resize(num*channel*width*height, _value);
			_s_data = &_data[0];
			if (train == phrase){
				_diff.resize(num*channel*width*height);
				_s_diff = &_diff[0];
			}
#endif

		}

		~blob(){
#if __PARALLELTYPE__ == __GPU__
			cuda_free<float_t>(_s_data);
			if(train == _phrase)
				cuda_free<float_t>(_s_diff);
#else
			vec_t().swap(_data);
			if (train == _phrase)
				vec_t().swap(_diff);
#endif
		}

		inline float_t* p_data(int n) {
			return _s_data + n*_cube_length;
		}

		inline float_t* p_diff(int n) {
			return _s_diff + n*_cube_length;
		}

		inline float_t* s_data(){ return _s_data; }

		inline float_t* s_diff(){ return _s_diff; }

		inline virtual const void _RESET_DATA() override
		{
#if __PARALLELTYPE__ == __GPU__
			cuda_setvalue<float_t>(_s_data,(float_t)(0),_num*_cube_length);
			if(train == _phrase)
				cuda_setvalue<float_t>(_s_diff,(float_t)(0),_num*_cube_length);
#else
			_data.resize(_num*_cube_length, float_t(0));
			if (train == _phrase)
				_diff.resize(_num*_cube_length, float_t(0));
#endif
		}

		inline virtual const int calculate_size() override{
			return test == _phrase ? _length*sizeof(float_t) : 2 * _length*sizeof(float_t); 
		}
		

	protected:

#if __PARALLELTYPE__ != __GPU__
		vec_t _data;

		vec_t _diff;

#endif

		float_t *_s_data;

		float_t *_s_diff;

	};
}
