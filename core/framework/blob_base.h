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

#include "../math/math_utils.h"
#include "../math/cuda/cuda_utils.h"
#include "../math/math_functions.h"

using namespace std;

namespace mycnn{

#define __BLOBMAXREC__ 0xFFFFFFFF

	class blob_base{

	public:

		blob_base(int num, int channel, int width, int height, phrase_type phrase , blob_type type){
			
			_width = width;
			_height = height;
			_channel = channel;
			_num = num;
			_channel_length = width*height;
			_cube_length = channel*width*height;
			_length = _num*_cube_length;
			_phrase = phrase;
			_blob_type = type;
			_s_data = NULL;
			_s_diff = NULL;
			_data_num = _num;
			_REC_ = 0;
			//check the input data dim
			long c_length_ = channel*width*height;
			CHECK_LT_OP(c_length_,_MAX_INT_,"BLOB cell length is out of bounds: %ld vs d", c_length_,_MAX_INT_);
			//check the input data dim
			long a_length = _num*_cube_length;
			CHECK_LT_OP(a_length,_MAX_INT_,"BLOB length is out of bounds: %ld vs d", a_length,_MAX_INT_);

		}

		virtual ~blob_base(){
#if __PARALLELTYPE__ == __CUDA__
			cuda_free(_s_data);
			if(train == _phrase)
				cuda_free(_s_diff);
#else
			free(_s_data);
			if (train == _phrase)
				free(_s_diff);
#endif
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

		inline int channel_length(){ return _channel_length; }

		inline int length(){ return _cube_length; }

		inline int count(){ return _length; }

		inline int data_num(){ return _data_num; }

		inline phrase_type phrase(){ return _phrase; }

		inline void set_data_num(int num){ _data_num = num; }

		virtual inline const int calculate_size() = 0;

		virtual const void _RESET_DATA() = 0;

		virtual const void _RESET_DIFF() = 0;

		virtual const void serializa(std::ostream& os) = 0;

		virtual const void load(std::ifstream& is) = 0;

		inline void _CHECK_SIZE_EQ(blob_base* blob_)
		{
			CHECK_EQ_OP(_channel,blob_->_channel,"_channel check does NOT match! ( %d vs %d )",_channel,blob_->_channel);
			CHECK_EQ_OP(_num,blob_->_num,"_num check does NOT match! ( %d vs %d )",_num,blob_->_num);
			CHECK_EQ_OP(_height,blob_->_height,"_height check does NOT match! ( %d vs %d )",_height,blob_->_height);
			CHECK_EQ_OP(_width,blob_->_width,"_width check does NOT match! ( %d vs %d )",_width,blob_->_width);
		}

		inline blob_type _TYPE(){return _blob_type;};

		inline bool is_output() {return _REC_ == 0;}

		inline void _REC(){
			if(_REC_ < __BLOBMAXREC__)
				_REC_ += 1;
			else
				LOG_WARNING("blob address %X has rec for %d times, but the max rec for cacu is %d",this,_REC_,__BLOBMAXREC__);
		}


	protected:		
		
		int _width;
		int _height;
		int _channel;
		int _num;
		int _cube_length;
		int _length;
		int _channel_length;

		int _data_num;
		phrase_type _phrase;

		inline void _SET_TYPE(blob_type blob_type_){_blob_type = blob_type_;};

		void *_s_data;

		void *_s_diff;

	private:
		
		blob_type _blob_type;

		unsigned _REC_;

	};
}
