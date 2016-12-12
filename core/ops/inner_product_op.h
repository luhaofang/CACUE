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


namespace mycnn{

	class inner_product_op : public operator_base
	{

	public:

		inner_product_op(blob *&data, args *&args_) : operator_base((blob_base*&)data, args_){
			check();

			o_blob = cacu_allocator::create_blob(data->num(), args_->at(0), 1, 1);

			_w = new weight("w", args_->at(0), data->channel(), data->width(), data->height(), data->phrase());
			_bias = new weight("bias", args_->at(0), 1, 1, 1, data->phrase());

			op();
		};

		~inner_product_op(){

			delete (blob *)o_blob;

			delete _w;
			delete _bias;
		};

		virtual const void check() override{
			//output_channel > 0
			CHECK_GT_OP(_args->at(0), 0);
			return;
		}

		virtual const void op() override {
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;

			for (int i = 0; i < s_blob_->num(); ++i){
				//inner_product
				cacu_sgemv(NOTRANS,_w->s_data(),_w->num(),s_blob_->p_data(i),s_blob_->length(),o_blob_->p_data(i));
				//elemwise_sum
				cacu_saxpby(o_blob_->p_data(i), (float_t)1, _bias->s_data(), (float_t)1, _args->at(0));
			}

			echo();
			return;
		}

		virtual const void grad(const solver_base *&solver_base) override{

		}

		virtual const void load(std::ifstream& is){
		
		}

		virtual const void save(std::ostream& os){
		
		}

		virtual const void echo()
		{

			LOG_INFO("%s:%d", "output_channel", (*_args)[0]);
			LOG_INFO("%f", ((blob*)o_blob)->s_data()[0]);
		}

	private:

		weight *_w;

		weight *_bias;
		
	};
};
