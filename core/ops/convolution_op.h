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
		convolution_op(blob *&data, args *&args_) : operator_base((blob_base *&)data, args_){

			check();

			int input_dim = data->width();
			int channel = data->channel();
			int num = data->num();
			int output_dim = (input_dim + 2 * args_->pad() - args_->kernel_size()) / args_->stride() + 1;
			o_blob = cacu_allocator::create_blob(num, args_->output_channel(), output_dim, output_dim);

			_w = new weight("w", args_->output_channel(), data->channel(), args_->kernel_size(), args_->kernel_size(), data->phrase());
			_bias = new weight("bias", args_->output_channel(), 1, 1, 1, data->phrase());

			_w->set_init_type(gaussian, 0.001f);
			_bias->set_init_type(constant);

#if __FFTW__ == ON
			//stride equal to 1
			if(args_->stride()==1){
				_fft_conv = new fft_conv(args_->kernel_size(),input_dim);
			}
			//if stride is not equal to 1, cancel fft convolution
			else {
				if (args_->pad() != 0)
					_padded_data = cacu_allocator::create_blob(num, data->channel(), input_dim + 2 * args_->pad(), input_dim + 2 * args_->pad());
				_col_data = cacu_allocator::create_blob(num, data->channel(), output_dim * args_->kernel_size(), output_dim*args_->kernel_size());
			}
#else
			if (args_->pad() != 0)
				_padded_data = cacu_allocator::create_blob(num, data->channel(), input_dim + 2 * args_->pad(), input_dim + 2 * args_->pad());
			_col_data = cacu_allocator::create_blob(num, data->channel(), output_dim * args_->kernel_size(), output_dim*args_->kernel_size());
#endif
		
		};

		~convolution_op(){
			delete (blob *)o_blob;		

			delete _w;
			delete _bias;

#if __FFTW__ == ON
			//stride equal to 1
			if(_args->stride()==1)
			{
				delete _fft_conv;
			}
			//if stride is not equal to 1, cancel fft convolution
			else{
				if (_args->pad() != 0)
					delete _padded_data;
				delete _col_data;
			}
#else
			if (_args->pad() != 0)
				delete _padded_data;
			delete _col_data;
#endif
		};

		virtual const void check() override{
			//output_channel > 0
			CHECK_GT_OP(_args->output_channel(), 0);
			//kernel_size > 0
			CHECK_GT_OP(_args->kernel_size(), 0);
			//stride > 0
			CHECK_GT_OP(_args->stride(), 0);
			return;
		}

		virtual const void op() override {
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;				

#if __FFTW__ == ON
			//stride equal to 1
			if(_args->stride()==1){
				//num
				for (int n = 0; n < s_blob_->num(); ++n){

				}
			}
			//if stride is not equal to 1, cancel fft convolution
			else
			{
				for (int i = 0; i < s_blob_->num(); ++i){
					if (_args->pad() != 0){
						cacu_padded_data<float_t>(s_blob_->p_data(i), s_blob_->channel(), s_blob_->width(), _args->pad(), _padded_data->p_data(i));
						cacu_img2col(_padded_data->p_data(i), _args->kernel_size(), _args->stride(), _padded_data->width(), s_blob_->channel(), o_blob_->width(), _col_data->p_data(i));
					}
					else
						cacu_img2col(s_blob_->p_data(i),_args->kernel_size(), _args->stride(), s_blob_->width(), s_blob_->channel(), o_blob_->width(), _col_data->p_data(i));
					cacu_sgemm(NOTRANS, TRANS, _w->s_data(), _w->num(), _w->length(), _col_data->p_data(i), o_blob_->width()*o_blob_->height(), o_blob_->p_data(i));
					cacu_ssxpy(_bias->s_data(), (float_t)(1), _bias->count(), o_blob_->p_data(i), (float_t)(1), o_blob_->length(), o_blob_->p_data(i));
				}
			}
#else			
			for (int i = 0; i < s_blob_->num(); ++i){
				if (_args->pad() != 0){
					cacu_padded_data<float_t>(s_blob_->p_data(i), s_blob_->channel(), s_blob_->width(), _args->at(3), _padded_data->p_data(i));
					cacu_img2col(_padded_data->p_data(i), _args->kernel_size(), _args->stride(), _padded_data->width(), s_blob_->channel(), o_blob_->width(), _col_data->p_data(i));
				}
				else
					cacu_img2col(s_blob_->p_data(i), _args->kernel_size(), _args->stride(), s_blob_->width(), s_blob_->channel(), o_blob_->width(), _col_data->p_data(i));
				cacu_sgemm(NOTRANS, TRANS, _w->s_data(), _w->num(), _w->length(), _col_data->p_data(i), o_blob_->width()*o_blob_->height(), o_blob_->p_data(i));
				cacu_ssxpy(_bias->s_data(), (float_t)(1), _bias->count(), o_blob_->p_data(i), (float_t)(1), o_blob_->length(), o_blob_->p_data(i));
			}
#endif
			echo();
			return;
		}

		virtual const void grad() override{
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;

			for (int i = 0; i < s_blob_->num(); ++i){

				//gradient propagation
				cacu_sgemm(TRANS,NOTRANS,o_blob_->p_diff(i),_args->output_channel(),o_blob_->width()*o_blob_->height(),_w->s_diff(),_w->length(),_col_data->p_diff(i));

				//weights gradient
				cacu_sgemm(NOTRANS,NOTRANS,o_blob_->p_diff(i),_args->output_channel(),o_blob_->width()*o_blob_->height(),_col_data->p_data(i),_w->length(),_w->s_diff());

				//bias gradient
				cacu_sumbysize(BYWIDTH,o_blob_->p_diff(i),o_blob_->length(),_bias->s_diff(),_bias->count());

				//col2img
				//unpadded
				if(_args->pad()!=0){
					cacu_col2img(_col_data->p_diff(i),_args->kernel_size(),_args->stride(),_args->input_dim() + 2 * _args->pad(),_args->channel(), o_blob_->width(),_padded_data->p_diff(i));
					cacu_unpadded_data(_padded_data->p_diff(i),_args->channel(),_args->input_dim() + 2 * _args->pad(),_args->pad(),s_blob_->p_diff(i));
				}
				else
					cacu_col2img(_col_data->p_diff(i),_args->kernel_size(),_args->stride(),_args->input_dim(),_args->channel(),o_blob_->width(),s_blob_->p_diff(i));
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

		inline void set_weight_init_type(param_init_type _type,float_t value = 0.0){_w->set_init_type(gaussian, value);}

		inline void set_bias_init_type(param_init_type _type,float_t value = 0.0){_bias->set_init_type(constant, value);}

	private:


		weight *_w;

		weight *_bias;

#if __FFTW__ == ON
		fft_conv *_fft_conv;
#endif
		blob *_padded_data;

		blob *_col_data;

	};
};
