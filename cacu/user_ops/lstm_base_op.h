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

#ifndef LSTM_BASE_OP_H_
#define LSTM_BASE_OP_H_

#include "../../tools/serializer_utils.h"

using namespace cacu_tools;

namespace cacu {

class lstm_base_op: public operator_base {

public:

	//output_channel, kernel_size, stride, pad, input_dim, channel
	lstm_base_op(blobs *&data, data_args *&args_) :
			operator_base(data, args_, CACU_CONV_TRANS) {
		_INIT_OP();
	}

	~lstm_base_op() {

	}

	void initial() override {
		int input_w = s_blobs->at(0)->width();
		int input_h= s_blobs->at(0)->height();
		int num = s_blobs->at(0)->num();

		int output_w, output_h;
		if(input_w != 1){
			output_w = input_w * _args->stride();
			_pad_w = ((input_w - 1) * _args->stride() + _args->kernel_size() - output_w);
			if(_pad_w % 2 != 0)
				_pad_w = _pad_w / 2 + 1;
			else
				_pad_w = _pad_w / 2;
		}
		else{
			output_w = (input_w - 1) * _args->stride()
							+ _args->kernel_size() - _args->pad() * 2;
			if (_args->kernel_size() == 1)
				output_w = input_w * _args->stride() - 2 * _args->pad();
			_pad_w = _args->pad();
		}

		if(input_h != 1){
			output_h = input_h * _args->stride();
			_pad_h = ((input_h - 1) * _args->stride() + _args->kernel_size() - output_h);
			if(_pad_h % 2 != 0)
				_pad_h = _pad_h / 2 + 1;
			else
				_pad_h = _pad_h / 2;
		}
		else{
			output_h = (input_h - 1) * _args->stride()
								+ _args->kernel_size() - _args->pad() * 2;
			if (_args->kernel_size() == 1)
				output_h = input_h * _args->stride() - 2 * _args->pad();
			_pad_h = _args->pad();
		}
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_oblobs();
			o_blobs = create_em_oblob(num, _args->output_channel(), output_w,
					output_h, _phase);

#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(num, _args->output_channel(), output_w, output_h, _phase));
#endif
			_col_data = create_opblob(1, _args->output_channel(),
					input_w * _args->kernel_size(),
					input_h * _args->kernel_size(), _phase);
			_bias_multiplier = create_opblob(1, 1, output_w, output_h, 1.0, _phase);
			_bias_multiplier->set_variable(false);
		} else {
			o_blobs->at(0)->resize(num, _args->output_channel(), output_w,
					output_h);
			_col_data->resize(1, _args->output_channel(),
					input_w * _args->kernel_size(),
					input_h * _args->kernel_size());
			_bias_multiplier->resize(1, 1, output_w, output_h);
			_bias_multiplier->set_data(1.0);
		}
	}

	void init_weights() override {
		_w = create_param("w", s_blobs->at(0)->channel(), _args->output_channel(),
				_args->kernel_size(), _args->kernel_size(), _phase);

		_bias = create_param("bias", _args->output_channel(), 1, 1, 1, _phase);
	}

	void check() override {
		if(_args == NULL)
			LOG_FATAL("deconvolution data args cannot equal to NULL!");
		//output_channel > 0
		CHECK_GT_OP(_args->output_channel(), 0, "output_channel must > 0 vs %d",
				_args->output_channel());
		//kernel_size > 0
		CHECK_GT_OP(_args->kernel_size(), 0, "kernel_size must > 0 vs %d",
				_args->kernel_size());
		//stride > 0
		CHECK_GT_OP(_args->stride(), 0, "stride must > 0 vs %d",
				_args->stride());
	}

	void op(blobs *&s_blobs_,blobs *&o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);
		blob *col_data_ = (blob*) _col_data;
		for (int i = 0; i < s_blob_->num(); ++i) {

			//forward propagation
			cacu_sgemm(NOTRANS, TRANS, _w->s_data(), _w->length(), _w->num(),
					s_blob_->p_data_d(i), o_blob_->width() * o_blob_->height(),
					1, col_data_->s_data(), 0);
			//col2img
			//unpadded
			cacu_col2img_pad(col_data_->s_data(), _args->kernel_size(),
					_args->stride(), s_blob->width(), s_blob->height(),
					s_blob->channel(), o_blob_->width(), o_blob_->height(),
					_args->pad(), _args->pad(), o_blob_->p_data_d(i));

			o_blob_->_sync(i);
		}
#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);
		blob *col_data_ = _col_data;
		blob *bias_multiplier = _bias_multiplier;

		col_offset = o_blob_->channel() / _group * _col_data->channel_length();
		w_offset = _w->count() / _group / _group;
		out_offset = _w->num() / _group * s_blob_->channel_length();

		for (int i = 0; i < s_blob_->num(); ++i) {
			//col_data_->blob_size();
			//gradient propagation
			for (int g = 0; g < _group; ++g)
			//cacu_sgemm(NOTRANS,TRANS, _w->s_data() + w_offset * g, _w->length() / _group, _w->num() / _group, o_blob_->p_diff(i) + out_offset * g, o_blob_->width() * o_blob_->height(), 1, col_data_->s_diff() + col_offset * g, 0);
				cacu_sgemm(NOTRANS,TRANS, s_blob_->p_data(i) + out_offset * g, s_blob_->width() * s_blob_->height(), _w->num() / _group, _w->s_data() + w_offset * g, _w->length() / _group, 1, col_data_->s_data() + col_offset * g, 0);
			//col2img
			//unpadded
			cacu_col2img_pad(col_data_->s_data(),_args->kernel_size(),_args->kernel_size(),_args->stride(),o_blob_->width(),o_blob_->height(),o_blob_->channel(),s_blob_->width(),s_blob_->height(),_pad_w,_pad_h, o_blob_->p_data(i));

			if(_is_use_bias)
			//cacu_sumbysize(BYWIDTH,o_blob_->p_diff(i),o_blob_->length(),1,_bias->s_diff(),1,o_blob_->width()*o_blob_->height());
				cacu_sgemm(NOTRANS, NOTRANS, bias_multiplier->s_data(), bias_multiplier->count(), 1, _bias->s_data(), _bias->count(),(float_t)(1),o_blob_->p_data(i),(float_t)(1));
		}
#endif

	}

	void grad(blobs *&s_blobs_,blobs *&o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);
		blob *col_data_ = (blob*) _col_data;
		for (int i = 0; i < s_blob_->num(); ++i) {
			//padded data if needed & img2col change
			cacu_img2col_pad(o_blob_->p_diff_d(i), _args->kernel_size(),
					_args->stride(), s_blob_->width(), s_blob_->height(),
					s_blob_->channel(), o_blob_->width(), o_blob_->height(),
					_args->pad(), _args->pad(), col_data_->s_diff());
			//backward convolution data
			cacu_sgemm(TRANS, NOTRANS, col_data_->s_diff(),
					s_blob_->width() * s_blob_->height(), _w->length(),
					_w->s_data(), _w->num(), (float_t) 1, s_blob_->p_diff_d(i),
					(float_t) 0);
			//add bias
			if (_is_use_bias)
				cacu_ssxpy(_bias->s_diff(), (float_t) (1), _bias->count(),
						s_blob_->p_diff_d(i), (float_t) (1), s_blob_->length(),
						s_blob_->p_diff_d(i));
			o_blob_->_sync(i);

			cacu_sgemm(NOTRANS, NOTRANS, col_data_->s_diff(), _w->length(),
					s_blob_->width() * s_blob_->height(), s_blob_->p_data_d(i),
					_w->num(), 1, _w->s_diff(), 1);
			//bias gradient
			if (_is_use_bias)
				cacu_sumbysize(BYWIDTH, s_blob_->p_diff(i), s_blob_->length(),
						1, _bias->s_diff(), 1,
						s_blob_->width() * s_blob_->height());
		}
#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);
		blob *col_data_ = _col_data;
		blob *bias_multiplier = _bias_multiplier;

		col_offset = o_blob_->channel() / _group * _col_data->channel_length();
		w_offset = _w->count() / _group / _group;
		out_offset = _w->num() / _group * s_blob_->channel_length();


		for (int i = 0; i < s_blob_->num(); ++i) {
			//padded data if needed & img2col change
			cacu_img2col_pad(o_blob_->p_diff(i), _args->kernel_size(),_args->kernel_size(), _args->stride(), o_blob_->width(), o_blob_->height(), o_blob_->channel(), s_blob_->width(), s_blob_->height(),_pad_w, _pad_h, col_data_->s_diff());

			//forward convolution data
			for (int g = 0; g < _group; ++g){
				cacu_sgemm(NOTRANS, NOTRANS, col_data_->s_diff() + col_offset * g, s_blob_->channel_length(),_w->length() / _group, _w->s_data() + w_offset * g, _w->num() / _group, (float_t)1, s_blob_->p_diff(i) + out_offset * g,(float_t)0);
				//weights gradient
				//cacu_img2col_pad(s_blob_->p_diff(i), _args->kernel_size(), _args->stride(),o_blob->width(),o_blob->height(),o_blob->channel(),s_blob_->width(),s_blob_->height(),_args->pad(),_args->pad(), col_data_->s_diff());
				cacu_sgemm(TRANS, NOTRANS, col_data_->s_diff() + col_offset * g, _w->length() / _group, s_blob_->channel_length(), s_blob_->p_data(i) + out_offset * g, _w->num() / _group, 1, _w->s_diff() + w_offset * g, 1);
				//cacu_bprint(_w,train);
			}
			//add bias
			if(_is_use_bias)
				cacu_sgemv(TRANS,o_blob_->p_diff(i),bias_multiplier->count(),bias_multiplier->s_data(),o_blob_->channel(),(float_t)(1),_bias->s_diff(),(float_t)(1));
			//cacu_ssxpy(_bias->s_data(), (float_t)(1), _bias->count(), o_blob_->p_data(i), (float_t)(1), o_blob_->length(), o_blob_->p_data(i));
		}
#endif
	}

	void load(std::ifstream& is) override {
		_w->load(is);
		if (_is_use_bias)
			_bias->load(is);
	}

	void save(std::ostream& os) override {
		_w->serializa(os);
		if (_is_use_bias)
			_bias->serializa(os);
	}

	void echo() override
	{
		LOG_INFO("create convolution transpose op:");
		LOG_INFO(
				"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d), kenrel_size: %d, stride: %d, pad: %d",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height(),
				_args->kernel_size(), _args->stride(), _args->pad());
	}

	inline void set_weight_init_type(param_init_type _type,
			float_t value = 0.0) {
		set_param_init_type(_type, _w, value);
	}

	inline void set_bias_init_type(param_init_type _type, float_t value = 0.0) {
		set_param_init_type(_type, _bias, value);
	}

	void set_is_use_bias(bool switcher_) {
		_is_use_bias = switcher_;
	}

	inline void set_group(int group) {
		CHECK_GT_OP(group, 0, "group must > 0 vs %d", group);
		CHECK_LE_OP(group, s_blobs->at(0)->channel(), "group must <= %d vs %d",
				_args->channel(), group);
		CHECK_EQ_OP(s_blobs->at(0)->channel() % group, 0,
				"channel mod group must == 0 vs %d", _args->channel() % group);
		LOG_INFO("group set: %d", group);
		this->_group = group;
	}

protected:

	bool _is_use_bias = true;

	weight *_w = NULL;

	weight *_bias = NULL;

	blob *_col_data = NULL;

	blob *_bias_multiplier = NULL;

	int _group = 1;


private:


	int _pad_w = 0;
	int _pad_h = 0;

	int col_offset = 0;

	int w_offset = 0;

	int out_offset = 0;

};
}


#endif
