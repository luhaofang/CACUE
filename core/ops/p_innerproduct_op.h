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

	class p_innerproduct_op : public operator_base
	{

	public:

		p_innerproduct_op(blob_base *&data, args *&args_) : operator_base(data, args_){
			check();

#if __USEMBEDDING__ == ON
			o_blob = create_em_oblob(data->num(), data->channel(), 1, 1, _phrase);
			_temp =  create_oblob(1, data->channel(), data->height(), data->width(), _phrase);
#else
			o_blob = create_oblob(data->num(), data->channel(), 1, 1, _phrase);
			_temp =  create_oblob(1, data->channel(), data->height(), data->width(), _phrase);
#endif
			_w = create_param("w", data->channel(), 1, data->width(), data->height(), _phrase);

			_bias = create_param("bias", data->channel(), 1, 1, 1, _phrase);
			_bias ->set_lr(2);

			echo();
		};

		~p_innerproduct_op(){
			delete _temp;
		};

		virtual const void check() override{
			return;
		}

		virtual const void op() override {

#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blob;
			blob *temp_ = (blob*)_temp;

			for(int i = 0; i < s_blob_->num(); ++i){
				cacu_copy(s_blob_->p_data(i),temp_->count(),temp_->s_data());
				cacu_ssx(_w->s_data(),_w->count(),temp_->s_data());
				cacu_sumbysize(BYWIDTH,temp_->s_data(),temp_->count(),1,o_blob_->p_data(i),0,s_blob_->height()*s_blob_->width());
				//bias added
				if(_is_use_bias)
				{
					cacu_ssxpy(_bias->s_data(),(float_t)(1),_bias->count(), o_blob_->p_data(i),(float_t)1,o_blob_->length(),o_blob_->p_data(i));
				}
				o_blob_->_sync(i);
			}
#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;
			blob *temp_ = (blob*)_temp;

			for(int i = 0; i < s_blob_->num(); ++i){
				cacu_copy(s_blob_->p_data(i),temp_->count(),temp_->s_data());
				cacu_ssx(_w->s_data(),_w->count(),temp_->s_data());
				cacu_sumbysize(BYWIDTH,temp_->s_data(),temp_->count(),1,o_blob_->p_data(i),0,s_blob_->height()*s_blob_->width());
				//bias added
				if(_is_use_bias)
				{
					cacu_ssxpy(_bias->s_data(),(float_t)(1),_bias->count(), o_blob_->p_data(i),(float_t)1,o_blob_->length(),o_blob_->p_data(i));
				}
			}
#endif
		}

		virtual const void grad() override{

#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blob;
			blob *temp_ = (blob*)_temp;

			for (int i = 0 ; i < s_blob_->num(); ++i){
				//gradient propagation
				cacu_cxsize(_w->s_data(), _w->count(), o_blob_->p_diff(i), o_blob_->length(), s_blob_->p_diff(i));
				//weights gradient
				cacu_cxsize(s_blob_->p_data(i), s_blob_->length(), o_blob_->p_diff(i), o_blob_->length(), temp_->s_diff());
				cacu_saxpby(temp_->s_diff(),(float_t)1,_w->s_diff(),(float_t)1,_w->count());
				if(_is_use_bias)
					//bias gradient
					cacu_saxpby(o_blob_->p_diff(i), 1, _bias->s_diff(), 1, o_blob_->count());
				s_blob_->_sync(i);
			}
#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;
			blob *temp_ = (blob*)_temp;

			for (int i = 0 ; i < s_blob_->num(); ++i){
				//gradient propagation
				cacu_cxsize(_w->s_data(), _w->count(), o_blob_->p_diff(i), o_blob_->length(), s_blob_->p_diff(i));
				//weights gradient
				cacu_cxsize(s_blob_->p_data(i), s_blob_->length(), o_blob_->p_diff(i), o_blob_->length(), temp_->s_diff());
				cacu_saxpby(temp_->s_diff(),(float_t)1,_w->s_diff(),(float_t)1,_w->count());
				if(_is_use_bias)
					//bias gradient
					cacu_saxpby(o_blob_->p_diff(i), 1, _bias->s_diff(), 1, o_blob_->count());
			}

#endif
		}

		virtual const void load(std::ifstream& is) override{
			_w->load(is);
			if(_is_use_bias)
				_bias->load(is);
		}

		virtual const void save(std::ostream& os) override{
			_w->serializa(os);
			if(_is_use_bias)
				_bias->serializa(os);
		}

		virtual const void echo() override
		{
			LOG_INFO("create p_inner_product op:");
			LOG_INFO("channel: %d, input_dim: %d, output_channel: %d, output_dim: %d",s_blob->channel(),s_blob->height(),o_blob->channel(),o_blob->height());
		}

		inline virtual const void LOOP_INIT_DATA_() override
		{
			o_blob->_RESET_DATA();

			_w->_RESET_DIFF();
			if(_is_use_bias)
				_bias->_RESET_DIFF();
		}

		inline void set_weight_init_type(param_init_type _type,float_t value = 0.0){set_param_init_type(_type, _w, value);}

		inline void set_bias_init_type(param_init_type _type,float_t value = 0.0){set_param_init_type(_type, _bias, value);}

		void is_use_bias(bool switcher_){ _is_use_bias = switcher_;};

	private:

		//p_innerproduct_op use bias switcher
		bool _is_use_bias = true;

		blob_base *_temp;

		weight *_w;

		weight *_bias;

	};
};
