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

#include "../../tools/random.h"
#include "../math/cuda/cuda_utils.h"

namespace mycnn{

	class weight : public blob{

	public:

		weight(chars_t name,int num,int channel,int width,int height,phrase_type phrase)
			: blob(num, channel, width, height,1,phrase){
			_name = name;
			_update_lr = 1.0;
			_decay_mult = 1.0;
		};

		inline chars_t name(){ return _name; }

		inline void set_lr(float_t lr_){ _update_lr = lr_; }

		inline float_t lr(){ return _update_lr; }

		inline void set_decay(float_t decay_mult_){_decay_mult = decay_mult_; }

		inline float_t decay(){ return _decay_mult; }

		~weight(){};

		void set_init_type(param_init_type type, float_t value = 0)
		{
			vec_t w(_length);
			switch (type)
			{
			case mycnn::constant:
				for (int i = 0; i < _length; ++i)
					w[i] = value;
				break;
			case mycnn::xavier:
				for (int i = 0; i < _length; ++i)
					w[i] = rand_t::urand(-value, value);
				break;
			case mycnn::gaussian:
				for (int i = 0; i < _length; ++i)
					w[i] = rand_t::gaussrand(value);
				break;
			case mycnn::msra:
				for (int i = 0; i < _length; ++i)
					w[i] = rand_t::gaussrand(value);
				break;
			default:
				break;
			}
			float_t* s_data_ = (float_t*)_s_data;
#if __PARALLELTYPE__ == __CUDA__
			cuda_copy2dev(s_data_,&w[0],_length);
			CUDA_CHECK(res);
#else
			for (int i = 0; i < _length; ++i)
				s_data_[i] = w[i];
#endif
			vec_t().swap(w);
		}


		/*
		 * serializa blob data, output data to model file
		 */
		inline void serializa(std::ostream& os)
		{
			float_t* s_data_ = (float_t*)_s_data;
	#if __PARALLELTYPE__ == __CUDA__
			os.write((char*)(&_length), sizeof(_length));
			vec_t _v(_length);
			cuda_copy2host(&_v[0],(float_t*)_s_data,_length);
			for (auto w : _v) os.write((char*)(&w), sizeof(w));
			vec_t().swap(_v);
	#else
			os.write((char*)(&_length), sizeof(_length));
			for(int i = 0 ; i < _length; ++i)
				os.write((char*)(&s_data_[i]), sizeof(s_data_[i]));
	#endif
		}

		/*
		 * loads blob data from model file
		 */
		inline void load(std::ifstream& is)
		{
			float_t* s_data_ = (float_t*)_s_data;
	#if __PARALLELTYPE__ == __CUDA__
			vec_t _v(_length);
			int length_;
			is.read(reinterpret_cast<char*>(&length_), sizeof(int));
			CHECK_EQ_OP(length_,_length,"parameter '%s' length is not equal to local weight: %d vs %d!", _name.c_str(), length_, _length);
			for (int i = 0; i < length_; i++){
				is.read(reinterpret_cast<char*>(&_v[i]), sizeof(float_t));
			}
			cuda_copy2dev((float_t*)_s_data, &_v[0],length_);
			vec_t().swap(_v);
	#else
			int length_;
			is.read(reinterpret_cast<char*>(&length_), sizeof(int));
			CHECK_EQ_OP(length_,_length,"parameter '%s' length is not equal to local weight: %d vs %d!", _name.c_str(), length_, _length);
			for (int i = 0; i < length_; i++){
				is.read(reinterpret_cast<char*>(s_data_ + i), sizeof(float_t));
			}
	#endif
		}

	private:

		chars_t _name;

		float_t _update_lr;

		float_t _decay_mult;

	};

};
