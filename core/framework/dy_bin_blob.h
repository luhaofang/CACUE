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


#include "../math/cuda/cuda_utils.h"

using namespace std;

namespace mycnn{

	class dy_bin_blob : public bin_blob{

	public:

		dy_bin_blob(int channel, int width, int height, unsigned int _value, phrase_type phrase)
			:bin_blob(1, channel, width, height, _value,  phrase){


		}

		~dy_bin_blob(){

		}

		inline const void copy_data_io(vec_i &data_ , int i = 0)
		{
			CHECK_EQ_OP(data_.size(),_cube_length,"blob size must be equal! %d vs %d",data_.size(),_cube_length);
#if __PARALLELTYPE__ == __GPU__
			cuda_copy2dev(p_data(i),&data_[0],_cube_length);
#else
			memcpy(p_data(i),&data_[0],_cube_length*sizeof(unsigned int));
#endif
		}

		inline const void copy_data_io(vec_t &data_, int i = 0)
		{
			CHECK_EQ_OP(data_.size(),_cube_length,"blob size must be equal! %d vs %d",data_.size(),_cube_length);
#if __PARALLELTYPE__ == __GPU__
			cuda_copy2dev(p_diff(i),&data_[0],_cube_length);
#else
			memcpy(p_diff(i),&data_[0],_cube_length*sizeof(float_t));
#endif
		}

	protected:

	};
}
