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

	template<typename DTYPE>
	class matrix{

	public:

		matrix(int width, int height){ 
			_data.resize(width*height, DTYPE(0));
			s_data = &_data[0];
			_width = width;
			_height = height;
			_length = width * height;
		}

		~matrix(){ 
			vector<DTYPE>().swap(_data);
		}

		inline DTYPE p_data(int x, int y) {
			int index = x*_width + y;
			CHECK_LT_OP(index,_length, "index must less than cube length %d vs %d",_length, index);
			return *(s_data + index);
		}

		inline int width(){ return _width; }

		inline int height(){ return _height; }

		void _RESET_DATA()
		{
			_data.resize(width*height, DTYPE(0));
		}

		int calculate_size(){ return _width*_height*sizeof(DTYPE) ;};

		DTYPE* s_data;

	private:

		vector<DTYPE> _data;

		int _width;
		int _height;

		int _length;
	};
}
