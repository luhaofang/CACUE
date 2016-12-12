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

//#include "math_utils.h"

using namespace std;

namespace mycnn{

	typedef struct{
		int x;
		int y;
		int z;
	}spatial3D;


	class blob_base{

	public:

		blob_base(int num, int channel, int width, int height, phrase_type phrase){
			
			_width = width;
			_height = height;
			_channel = channel;
			_num = num;
			_cube_length = channel*width*height;
			_length = _num*_cube_length;
			_phrase = phrase;
		}

		~blob_base(){

		}		
		
		inline int index(int c, int x, int y)
		{
			return c*_cube_length + x*_width + y;
		}

		inline const spatial3D index3D(int i)
		{
			spatial3D s = { i / _cube_length, (i % _cube_length) / _width, (i % _cube_length) % _width };
			return s;
		}

		inline int num(){ return _num; }

		inline int channel(){ return _channel; }

		inline int width(){ return _width; }

		inline int height(){ return _height; }

		inline int length(){ return _cube_length; }

		inline int count(){ return _length; }

		inline phrase_type phrase(){ return _phrase; }

		inline virtual const int calculate_size() = 0;

		inline virtual const void _RESET_DATA() = 0;
		

	protected:		
		
		int _width;
		int _height;
		int _channel;
		int _num;
		int _cube_length;
		int _length;
		phrase_type _phrase;

	private:

		
	};
}