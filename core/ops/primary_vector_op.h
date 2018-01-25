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

	class primary_vector_op : public operator_base
	{

	public:

		primary_vector_op(blob_base *&data, args *&args_) : operator_base(data, args_, CACU_PRIMARY_VECTOR){
			check();
			initial(data,args_);
			init_weights(data,args_);
			echo();

		};

		~primary_vector_op(){

			delete _bias_multiplier;
		};

		virtual const void initial(blob_base *&data, args *&args_) override{
#if __USEMBEDDING__ == ON
			o_blob = create_em_oblob(data->num(), _args->output_channel(), 1, 1, _phrase);
#else
			o_blob = create_oblob(data->num(), _args->output_channel(), 1, 1, _phrase);
#endif

			_bias_multiplier = cacu_allocator::create_blob(1, data->num(), 1, 1, (float_t)(1), _phrase);
		}

		virtual const void init_weights(blob_base *&data, args *&args_) override{

			_w = create_param("w", _args->output_channel(), data->channel(), data->width(), data->height(), _phrase);

			_bias = create_param("bias", _args->output_channel(), 1, 1, 1, _phrase);
			_bias ->set_lr(2);
		}

		virtual const void check() override{
			//output_channel > 0
			CHECK_GT_OP(_args->output_channel(), 0,"output_channel must > 0 vs %d",_args->output_channel());
		}

		virtual const void op() override {

			blob *bias_multiplier = (blob*)_bias_multiplier;

#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blob;

			for(int i = 0; i < s_blob_->num(); ++i){
				cacu_sgemm(TRANS, NOTRANS, _w->s_data(),_w->num(), _w->length(),s_blob_->p_data_d(i),1, 1 ,o_blob_->p_data_d(i),0);
				//bias added
				if(_is_use_bias)
					cacu_saxpby(_bias->s_data(),(float_t)(1),o_blob_->p_data_d(i),(float_t)(1),_bias->count());
					//cacu_ssxpy(_bias->s_data(),(float_t)(1),_bias->count(), o_blob_->p_data_d(i),(float_t)1,o_blob_->length(),o_blob_->p_data_d(i));
				o_blob_->_sync(i);
			}
#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;

			cacu_sgemm(TRANS, NOTRANS, _w->s_data(),_w->num(), _w->length(),s_blob_->s_data(),s_blob_->num(), 1 ,o_blob_->s_data(),0);
			//bias added
			if(_is_use_bias)
				for(int i = 0 ; i < s_blob_->num(); ++i)
				{
					cacu_saxpby(_bias->s_data(),(float_t)(1),o_blob_->p_data(i),(float_t)(1),_bias->count());
					//cacu_ssxpy(_bias->s_data(),(float_t)(1),_bias->count(), o_blob_->p_data(i),(float_t)1,o_blob_->length(),o_blob_->p_data(i));
				}
#endif
		}

		virtual const void grad() override{

			blob *bias_multiplier = (blob*)_bias_multiplier;

#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blob;

			for(int i = 0; i < s_blob_->num();++i){
				//gradient propagation
				cacu_sgemm(NOTRANS,NOTRANS,_w->s_data(),_w->length(),_w->num(), o_blob_->p_diff_d(i), 1, 1 ,s_blob_->p_diff_d(i), 0);
				//weights gradient
				cacu_sgemm(NOTRANS,TRANS,s_blob_->p_data_d(i), s_blob_->length(), 1, o_blob_->p_diff_d(i), o_blob_->length(),1,_w->s_diff(),1);
				if(_is_use_bias)
					//bias gradient
					cacu_sumbysize(BYHEIGHT,o_blob_->p_diff_d(i),o_blob_->length(),1 ,_bias->s_diff(),1,_bias->count());
				s_blob_->_sync(i);
			}
#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;

			//gradient propagation
			cacu_sgemm(NOTRANS,NOTRANS,_w->s_data(),_w->length(),_w->num(), o_blob_->s_diff(), o_blob_->num(), 1 ,s_blob_->s_diff(), 0);
			//weights gradient
			cacu_sgemm(NOTRANS,TRANS,s_blob_->s_data(), s_blob_->length(), o_blob_->num(), o_blob_->s_diff(), o_blob_->length(),1,_w->s_diff(),1);
			if(_is_use_bias)
				//bias gradient
				cacu_sgemv(NOTRANS,o_blob_->s_diff(),o_blob_->channel(),bias_multiplier->s_data(),bias_multiplier->count(),(float_t)(1),_bias->s_diff(),(float_t)(1));
				//cacu_sumbysize(BYHEIGHT,o_blob_->s_diff(),o_blob_->count(),1 ,_bias->s_diff(),1,_bias->count());
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
			LOG_INFO("create inner_product op:");
			LOG_INFO("channel: %d, input_dim: %d, output_channel: %d, output_dim: %d",s_blob->channel(),s_blob->height(),o_blob->channel(),o_blob->height());
		}

		inline virtual const void LOOP_INIT_DATA_() override
		{
			o_blob->_RESET_DATA();

			_w->_RESET_DIFF();
			if(_is_use_bias)
				_bias->_RESET_DIFF();
		}

		inline virtual const void set_phrase(phrase_type phrase_) override {
			_phrase = phrase_;
		}

		inline void set_weight_init_type(param_init_type _type,float_t value = 0.0){set_param_init_type(_type, _w, value);}

		inline void set_bias_init_type(param_init_type _type,float_t value = 0.0){set_param_init_type(_type, _bias, value);}

		void is_use_bias(bool switcher_){ _is_use_bias = switcher_;};

	private:

		//inner_product_op use bias switcher
		bool _is_use_bias = true;

		weight *_w;

		weight *_bias;

		blob_base *_bias_multiplier;

	};
};
