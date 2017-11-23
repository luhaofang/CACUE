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

	class deconvolution_op : public operator_base
	{

	public:

		//output_channel, kernel_size, stride, pad, input_dim, channel
		deconvolution_op(blob_base *&data, args *&args_) : operator_base(data, args_, CACU_DECONVOLUTION){

			check();
			int input_dim = data->width();
			int channel = data->channel();
			int num = data->num();

			int output_dim = (input_dim - 1) * _args->stride() + _args->kernel_size() - _args->pad() * 2;
			if(_args->kernel_size() == 1)
				output_dim = input_dim * _args->stride() - 2 * _args->pad();
#if __USEMBEDDING__ == ON
			o_blob = create_em_oblob(num, _args->output_channel(), output_dim, output_dim, _phrase);

#else
			o_blob = create_oblob(num, _args->output_channel(), output_dim, output_dim, _phrase);
#endif
			_w = create_param("w", data->channel(), _args->output_channel(), _args->kernel_size(), _args->kernel_size(), _phrase);

			_bias = create_param("bias", data->channel(), 1, 1, 1, _phrase);
			_bias ->set_lr(2.0);

			_col_data = cacu_allocator::create_blob(1, _args->output_channel(), input_dim * _args->kernel_size(), input_dim*_args->kernel_size(), _phrase);
			echo();
		};

		~deconvolution_op(){

			delete _col_data;
		};

		virtual const void check() override{
			//output_channel > 0
			CHECK_GT_OP(_args->output_channel(), 0,"output_channel must > 0 vs %d",_args->output_channel());
			//kernel_size > 0
			CHECK_GT_OP(_args->kernel_size(), 0,"kernel_size must > 0 vs %d",_args->kernel_size());
			//stride > 0
			CHECK_GT_OP(_args->stride(), 0,"stride must > 0 vs %d",_args->stride());
		}

		virtual const void op() override {

#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blob;
			blob *col_data_ = (blob*)_col_data;
			for (int i = 0; i < s_blob_->num(); ++i){

				//forward propagation
				cacu_sgemm(NOTRANS,TRANS,_w->s_data(),_w->length(),_w->num(),s_blob_->p_data_d(i),o_blob_->width()*o_blob_->height(),1 ,col_data_->s_data(),0);
				//col2img
				//unpadded
				cacu_col2img_pad(col_data_->s_data(),_args->kernel_size(),_args->stride(),o_blob_->height(),o_blob_->channel(),s_blob_->width(),_args->pad(), o_blob_->p_data_d(i));

				o_blob_->_sync(i);
			}
#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;
			blob *col_data_ = (blob*)_col_data;

			for (int i = 0; i < s_blob_->num(); ++i){

				//forward propagation
				cacu_sgemm(NOTRANS,TRANS,_w->s_data(),_w->length(),_w->num(),s_blob_->p_data(i),o_blob_->width()*o_blob_->height(),1 ,col_data_->s_data(),0);
				//col2img
				//unpadded
				cacu_col2img_pad(col_data_->s_data(),_args->kernel_size(),_args->stride(),o_blob_->height(),o_blob_->channel(),s_blob_->width(),_args->pad(), o_blob_->p_data(i));
			}
#endif

		}

		virtual const void grad() override{

#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blob;
			blob *col_data_ = (blob*)_col_data;
			for (int i = 0; i < s_blob_->num(); ++i){
				//padded data if needed & img2col change
				cacu_img2col_pad(o_blob_->p_diff_d(i), _args->kernel_size(), _args->stride(), o_blob_->width(), o_blob_->channel(), s_blob_->width(), _args->pad(), col_data_->s_diff());
				//backward convolution data
				cacu_sgemm(TRANS, NOTRANS, col_data_->s_diff(), s_blob_->width()*s_blob_->height(),_w->length(), _w->s_data(),_w->num(), (float_t)1,s_blob_->p_diff_d(i),(float_t)0);
				//add bias
				if(_is_use_bias)
					cacu_ssxpy(_bias->s_diff(), (float_t)(1), _bias->count(), s_blob_->p_diff_d(i), (float_t)(1), s_blob_->length(), s_blob_->p_diff_d(i));
				o_blob_->_sync(i);

				cacu_sgemm(NOTRANS,NOTRANS,col_data_->s_diff(),_w->length(),s_blob_->width()*s_blob_->height(),s_blob_->p_data_d(i),_w->num(),1,_w->s_diff(),1);
				//bias gradient
				if(_is_use_bias)
					cacu_sumbysize(BYWIDTH,s_blob_->p_diff(i),s_blob_->length(),1,_bias->s_diff(),1,s_blob_->width()*s_blob_->height());
			}
#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;
			blob *col_data_ = (blob*)_col_data;
			for (int i = 0; i < s_blob_->num(); ++i){
				//padded data if needed & img2col change
				cacu_img2col_pad(o_blob_->p_diff(i), _args->kernel_size(), _args->stride(), o_blob_->width(), o_blob_->channel(), s_blob_->width(), _args->pad(), col_data_->s_diff());
				//backward convolution data
				cacu_sgemm(TRANS, NOTRANS, col_data_->s_diff(), s_blob_->width()*s_blob_->height(),_w->length(), _w->s_data(),_w->num(), (float_t)1,s_blob_->p_diff(i),(float_t)0);
				//add bias
				if(_is_use_bias)
					cacu_ssxpy(_bias->s_diff(), (float_t)(1), _bias->count(), s_blob_->p_diff(i), (float_t)(1), s_blob_->length(), s_blob_->p_diff(i));

				cacu_sgemm(NOTRANS,NOTRANS,col_data_->s_diff(),_w->length(),s_blob_->width()*s_blob_->height(),s_blob_->p_data(i),_w->num(),1,_w->s_diff(),1);
				//bias gradient
				if(_is_use_bias)
					cacu_sumbysize(BYWIDTH,s_blob_->p_diff(i),s_blob_->length(),1,_bias->s_diff(),1,s_blob_->width()*s_blob_->height());

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
			LOG_INFO("create deconvolution op:");
			LOG_INFO("channel: %d, input_dim: %d, output_channel: %d, output_dim: %d, kenrel_size: %d, stride: %d, pad: %d",s_blob->channel(),s_blob->height(),o_blob->channel(),o_blob->height(), _args->kernel_size(),_args->stride(),_args->pad());
		}

		inline virtual const void LOOP_INIT_DATA_() override
		{
			o_blob->_RESET_DATA();
			_w->_RESET_DIFF();
			if(_is_use_bias)
				_bias->_RESET_DIFF();
			_col_data->_RESET_DATA();
		}

		inline virtual const void set_phrase(phrase_type phrase_) override {
			_phrase = phrase_;
		}

		inline void set_weight_init_type(param_init_type _type,float_t value = 0.0){set_param_init_type(_type, _w, value);}

		inline void set_bias_init_type(param_init_type _type,float_t value = 0.0){set_param_init_type(_type, _bias, value);}

	    void set_is_use_bias(bool switcher_){
	    	_is_use_bias = switcher_;
	    };

	protected:

		bool _is_use_bias = true;

		weight *_w;

		weight *_bias;

		blob_base *_col_data;

	};
};
