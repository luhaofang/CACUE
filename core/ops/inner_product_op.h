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

		inner_product_op(blob_base *&data, args *&args_) : operator_base(data, args_){
			check();

			o_blob = cacu_allocator::create_blob(data->num(), _args->output_channel(), 1, 1, _phrase);

			_w = create_param("w", _args->output_channel(), data->channel(), data->width(), data->height(), _phrase);
			if(_is_use_bias){
				_bias = create_param("bias", _args->output_channel(), 1, 1, 1, _phrase);
				_bias ->set_lr(2);
			}
			echo();
		};

		~inner_product_op(){

		};

		virtual const void check() override{
			//output_channel > 0
			CHECK_GT_OP(_args->output_channel(), 0,"output_channel must > 0 vs %d",_args->output_channel());
		}

		virtual const void op() override {
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;

			cacu_sgemm(TRANS, NOTRANS, _w->s_data(),_w->num(), _w->length(),s_blob_->s_data(),s_blob_->num(), 1 ,o_blob_->s_data(),0);
			//bias added
			if(_is_use_bias)
				for(int i = 0 ; i < s_blob_->num(); ++i)
				{
					cacu_ssxpy(_bias->s_data(),(float_t)(1),_bias->count(), o_blob_->p_data(i),(float_t)1,o_blob_->length(),o_blob_->p_data(i));
				}
		}

		virtual const void grad() override{
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;

			//gradient propagation
			cacu_sgemm(NOTRANS,NOTRANS,_w->s_data(),_w->length(),_w->num(), o_blob_->s_diff(), o_blob_->num(), 1 ,s_blob_->s_diff(), 0);
			//weights gradient
			cacu_sgemm(NOTRANS,TRANS,s_blob_->s_data(), s_blob_->length(), o_blob_->num(), o_blob_->s_diff(), o_blob_->length(),1,_w->s_diff(),1);
			if(_is_use_bias)
				//bias gradient
				cacu_sumbysize(BYHEIGHT,o_blob_->s_diff(),o_blob_->count(),1 ,_bias->s_diff(),1,_bias->count());

		}

		virtual const void load(std::ifstream& is) override{
			_w->loads(is);
			if(_is_use_bias)
				_bias->loads(is);
		}

		virtual const void save(std::ostream& os) override{
			_w->serializa(os);
			if(_is_use_bias)
				_bias->serializa(os);
		}

		virtual const void echo() override
		{
			LOG_INFO("create inner_product op:");
			LOG_INFO("channel: %d, output_channel: %d",s_blob->channel(),o_blob->channel());
		}

		inline virtual const void LOOP_INIT_DATA_() override
		{
			o_blob->_RESET_DATA();

			_w->_RESET_DIFF();
			_bias->_RESET_DIFF();
		}

		inline void set_weight_init_type(param_init_type _type,float_t value = 0.0){set_param_init_type(_type, _w, value);}

		inline void set_bias_init_type(param_init_type _type,float_t value = 0.0){set_param_init_type(_type, _bias, value);}

		void is_use_bias(bool switcher_){ _is_use_bias = switcher_;};

	private:

		//inner_product_op use bias switcher
		bool _is_use_bias = true;

		weight *_w;

		weight *_bias;
		
	};
};
