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

namespace mycnn{

	class dropout_op : public operator_base
	{

	public:

		dropout_op(blob_base *&data, args *&args_) : operator_base(data, args_){
			check();

			int input_dim = data->width();
			int channel = data->channel();
			int num = data->num();

			o_blob = cacu_allocator::create_blob(num, channel, input_dim, input_dim, _phrase);
			_rand_vect = cacu_allocator::create_blob(num,channel,input_dim,input_dim, test);

		};

		~dropout_op(){
			delete (blob*)o_blob;
			delete _rand_vect;
		};

		virtual const void check() override{
			return;
		}

		virtual const void op() override {
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;
			if (test == o_blob_->phrase())
				cacu_copy(s_blob_->s_data(), s_blob_->count(),o_blob_->s_data());
			else
			{
				rand_vector(_rand_vect->s_data(),_rand_vect->count(),_ratio);
				cacu_ssx(_rand_vect->s_data(), o_blob_->count(), o_blob_->s_data());
			}
		}

		virtual const void grad() override{

			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;
			if (test == o_blob_->phrase())
				cacu_copy(o_blob_->s_diff(), s_blob_->count(),s_blob_->s_diff());
			else
			{
				//one of the dropout's implementation
				cacu_ssx(_rand_vect->s_data(), s_blob_->count(), s_blob_->s_diff());
				cacu_copy(o_blob_->s_diff(),o_blob_->count(),s_blob_->s_diff());
				//ratio's scale implementation
				cacu_scalex(s_blob_->s_diff(),o_blob_->count(),_ratio);
			}
		}

		virtual const void load(std::ifstream& is) override{

		}

		virtual const void save(std::ostream& os) override{

		}

		virtual const void echo() override
		{
			//LOG_INFO("%f", ((blob*)o_blob)->s_data()[0]);
		}

		inline virtual const void LOOP_INIT_DATA_() override
		{
			blob *o_blob_ = (blob*)o_blob;
			o_blob_->_RESET_DATA();
			_rand_vect->_RESET_DATA();
		}

		float_t _ratio = 0.5f;

	private:

		blob *_rand_vect;

	};
};
