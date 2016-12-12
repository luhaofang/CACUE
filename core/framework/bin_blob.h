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


using namespace std;

namespace mycnn{

	class bin_blob : public blob_base{

	public:

		bin_blob(int num, int channel, int width, int height, unsigned int _value = 0, phrase_type phrase = test)
			:blob_base(num, channel, width, height, phrase){
			_data.resize(num * channel * width * height, _value);
			_s_data = &_data[0];
			if (train == phrase){
				_diff.resize(num*channel*width*height);
				_s_diff = &_diff[0];
			}
		}

		~bin_blob(){
			vec_i().swap(_data);
			if (train == _phrase)
				vec_t().swap(_diff);
		}

		inline unsigned int* p_data(int n) {
			return _s_data + n*_cube_length;
		}

		inline float_t* p_diff(int n) {
			return _s_diff + n*_cube_length;
		}

		inline unsigned int* s_data(){ return _s_data; }

		inline float_t* s_diff(){ return _s_diff; }

		inline virtual const void _RESET_DATA() override
		{
			_data.resize(_num*_cube_length, 0);
			if (train == _phrase)
				_diff.resize(_num*_cube_length, float_t(0));
		}

		inline virtual const int calculate_size() override{
			return test == _phrase ? _length*sizeof(unsigned int) : _length*sizeof(unsigned int) + _length*sizeof(float_t);
		}


	protected:

		vec_i _data;

		vec_t _diff;

		unsigned int *_s_data;

		float_t *_s_diff;

	};
}