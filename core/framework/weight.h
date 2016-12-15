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

		weight(chars_t name,int num,int channel,int width,int height,phrase_type phrase = test)
			: blob(num, channel, width, height,1,phrase){
			_name = name;
			_update_lr = 1.0;
		};

		inline chars_t name(){ return _name; }

		inline void set_lr(float_t lr){ _update_lr = lr; }

		inline float_t get_lr(float_t lr){ return _update_lr; }

		~weight(){};

		void set_init_type(param_init_type type, float_t value = 0)
		{
			rand_t *r = new rand_t();
			vec_t w(_length);
			switch (type)
			{
			case mycnn::constant:
				for (int i = 0; i < _length; i++)
					w[i] = value;
				break;
			case mycnn::xavier:
				for (int i = 0; i < _length; i++)
					w[i] = r->frand(-value, value);
				break;
			case mycnn::gaussian:
				for (int i = 0; i < _length; i++)
					w[i] = r->gaussrand(value);
				break;
			case mycnn::msra:
				for (int i = 0; i < _length; i++)
					w[i] = r->gaussrand(value);
				break;
			default:
				break;
			}
#if __PARALLELTYPE__ == __GPU__
			cuda_copy2dev(_s_data,&w[0],_length);
			CUDA_CHECK(res);
#else
			for (int i = 0; i < _length; i++)
				_s_data[i] = w[i];
#endif
			delete r;
			vec_t().swap(w);
		}

	private:

		chars_t _name;

		float_t _update_lr;

	};

};
