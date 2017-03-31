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

	class convolution_op : public operator_base
	{

	public:

		//output_channel, kernel_size, stride, pad, input_dim, channel
		convolution_op(blob_base *&data, args *&args_) : operator_base(data, args_){

			check();
			int input_dim = data->width();
			int channel = data->channel();
			int num = data->num();
			int output_dim = (input_dim + 2 * _args->pad() - _args->kernel_size()) / _args->stride() + 1;
			o_blob = cacu_allocator::create_blob(num, _args->output_channel(), output_dim, output_dim, _phrase);

			_w = create_param("w", _args->output_channel(), data->channel(), _args->kernel_size(), _args->kernel_size(), _phrase);
			if(_is_use_bias){
				_bias = create_param("bias", _args->output_channel(), 1, 1, 1, _phrase);
				_bias ->set_lr(2);
			}

			if (_args->pad() != 0)
				_padded_data = cacu_allocator::create_blob(num, data->channel(), input_dim + 2 * _args->pad(), input_dim + 2 * _args->pad(), _phrase);
			_col_data = cacu_allocator::create_blob(num, data->channel(), output_dim * _args->kernel_size(), output_dim*_args->kernel_size(), _phrase);
			echo();
		};

		~convolution_op(){

			if (_args->pad() != 0)
				delete _padded_data;
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
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;

			for (int i = 0; i < s_blob_->num(); ++i){
				//padded data if needed & img2col change
				if (_args->pad() != 0){
					cacu_padded_data(s_blob_->p_data(i), s_blob_->channel(), s_blob_->width(), _args->pad(), _padded_data->p_data(i));
					cacu_img2col(_padded_data->p_data(i), _args->kernel_size(), _args->stride(), _padded_data->width(), s_blob_->channel(), o_blob_->width(), _col_data->p_data(i));
				}
				else
					cacu_img2col(s_blob_->p_data(i), _args->kernel_size(), _args->stride(), s_blob_->width(), s_blob_->channel(), o_blob_->width(), _col_data->p_data(i));
				//forward convolution data
				cacu_sgemm(TRANS, NOTRANS, _col_data->p_data(i), o_blob_->width()*o_blob_->height(),_w->length(), _w->s_data(),_w->num(), (float_t)1,o_blob_->p_data(i),(float_t)0);
				//add bias
				if(_is_use_bias)
					cacu_ssxpy(_bias->s_data(), (float_t)(1), _bias->count(), o_blob_->p_data(i), (float_t)(1), o_blob_->length(), o_blob_->p_data(i));
			}
		}

		virtual const void grad() override{
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;

			for (int i = 0; i < s_blob_->num(); ++i){
				//gradient propagation
				cacu_sgemm(NOTRANS,TRANS,_w->s_data(),_w->length(),_w->num(),o_blob_->p_diff(i),o_blob_->width()*o_blob_->height(),1 ,_col_data->p_diff(i),0);
				//col2img
				//unpadded
				if(_args->pad() != 0){
					cacu_col2img(_col_data->p_diff(i), _args->kernel_size(),_args->stride(), _padded_data->width(), _args->channel(), o_blob_->width(), _padded_data->p_diff(i));
					cacu_unpadded_data(_padded_data->p_diff(i), _padded_data->channel(), _padded_data->width(), _args->pad(),s_blob_->p_diff(i));
				}
				else
					cacu_col2img(_col_data->p_diff(i),_args->kernel_size(),_args->stride(),_args->input_dim(),_args->channel(),o_blob_->width(),s_blob_->p_diff(i));
				//weights gradient
				cacu_sgemm(NOTRANS,NOTRANS,_col_data->p_data(i),_w->length(),o_blob_->width()*o_blob_->height(),o_blob_->p_diff(i),_w->num(),1,_w->s_diff(),1);
				//bias gradient
				if(_is_use_bias)
					cacu_sumbysize(BYWIDTH,o_blob_->p_diff(i),o_blob_->length(),1,_bias->s_diff(),1,o_blob_->width()*o_blob_->height());

			}
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
			LOG_INFO("create convolution op:");
			LOG_INFO("channel: %d, input_dim: %d, output_channel: %d, output_dim: %d",s_blob->channel(),s_blob->height(),o_blob->channel(),o_blob->height());
		}

		inline virtual const void LOOP_INIT_DATA_() override
		{
			o_blob->_RESET_DATA();
			_w->_RESET_DIFF();
			_bias->_RESET_DIFF();

			if (_args->pad() != 0)
				_padded_data->_RESET_DATA();
			_col_data->_RESET_DATA();
		}

		inline void set_weight_init_type(param_init_type _type,float_t value = 0.0){set_param_init_type(_type, _w, value);}

		inline void set_bias_init_type(param_init_type _type,float_t value = 0.0){set_param_init_type(_type, _bias, value);}

		void is_use_bias(bool switcher_){ _is_use_bias = switcher_;};

	private:

		bool _is_use_bias = true;

		weight *_w;

		weight *_bias;

		blob *_padded_data;

		blob *_col_data;

	};
};
