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

namespace cacu {

class convolution_op: public operator_base {

public:

	//output_channel, kernel_size, stride, pad, input_dim, channel
	convolution_op(blob_base *&data, data_args *&args_) :
			operator_base(data, args_, CACU_CONVOLUTION) {

		check();
		initial();
		init_weights();
		echo();
	}

	~convolution_op() {

	}

	virtual const void initial() override {

		int output_w = (s_blob->width() + 2 * _args->pad()
				- _args->kernel_size()) / _args->stride() + 1;
		if (_args->kernel_size() == 1)
			output_w = (s_blob->width() + 2 * _args->pad()) / _args->stride();

		int output_h = (s_blob->height() + 2 * _args->pad()
				- _args->kernel_size()) / _args->stride() + 1;
		if (_args->kernel_size() == 1)
			output_h = (s_blob->height() + 2 * _args->pad()) / _args->stride();
		if (o_blob == NULL) {
#if __USEMBEDDING__ == ON
			o_blob = create_em_oblob(s_blob->num(), _args->output_channel(),
					output_w, output_h, _phase);

#else
			o_blob = create_oblob(s_blob->num(), _args->output_channel(), output_w, output_h, _phase);
#endif
			_col_data = create_opblob(1, s_blob->channel(),
					output_w * _args->kernel_size(),
					output_h * _args->kernel_size(), _phase);
			_bias_multiplier = create_opblob(1, 1, output_w, output_h,
					(float_t) (1), _phase);
		} else {

			o_blob->resize(s_blob->num(), _args->output_channel(), output_w,
					output_h);
			_col_data->resize(1, s_blob->channel(),
					output_w * _args->kernel_size(),
					output_h * _args->kernel_size());
			_bias_multiplier->resize(1, 1, output_w, output_h, (float_t) (1));
		}
	}

	virtual const void init_weights() override {
		_w = create_param("w", _args->output_channel(), s_blob->channel(),
				_args->kernel_size(), _args->kernel_size(), _phase);

		_bias = create_param("bias", _args->output_channel(), 1, 1, 1, _phase);
		_bias->set_lr(2);
	}

	virtual const void check() override {
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

	virtual const void op() override {

		col_offset = s_blob->channel() / _group * _col_data->channel_length();
		w_offset = _w->count() / _group / _group;
		out_offset = _w->num() / _group * o_blob->channel_length();

		blob *col_data_ = (blob*) _col_data;
		blob *bias_multiplier = (blob*) _bias_multiplier;

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;

		for (int i = 0; i < s_blob_->num(); ++i) {
			//padded data if needed & img2col change
			cacu_img2col_pad(s_blob_->p_data_d(i), _args->kernel_size(),
					_args->stride(), s_blob_->width(), s_blob_->height(),
					s_blob_->channel(), o_blob_->width(), o_blob_->height(),
					_args->pad(), _args->pad(), col_data_->s_data());
			//forward convolution data
			for (int g = 0; g < _group; ++g)
				cacu_sgemm(NOTRANS, NOTRANS,
						col_data_->s_data() + col_offset * g,
						o_blob_->channel_length(), _w->length() / _group,
						_w->s_data() + w_offset * g, _w->num() / _group,
						(float_t) 1, o_blob_->p_data_d(i) + out_offset * g,
						(float_t) 0);
			//add bias
			if (_is_use_bias)
				cacu_sgemm(NOTRANS, NOTRANS, bias_multiplier->s_data(),
						bias_multiplier->count(), 1, _bias->s_data(),
						_bias->count(), (float_t) (1), o_blob_->p_data_d(i),
						(float_t) (1));
			//cacu_ssxpy(_bias->s_data(), (float_t)(1), _bias->count(), o_blob_->p_data_d(i), (float_t)(1), o_blob_->length(), o_blob_->p_data_d(i));
			o_blob_->_sync(i);
		}
#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blob;

		for (int i = 0; i < s_blob_->num(); ++i) {
			//padded data if needed & img2col change
			cacu_img2col_pad(s_blob_->p_data(i), _args->kernel_size(), _args->stride(), s_blob_->width(), s_blob_->height(), s_blob_->channel(), o_blob_->width(), o_blob_->height(),_args->pad(), _args->pad(), col_data_->s_data());
			//mycnn_tools::cacu_output(col_data_->s_data(),col_data_->count(),"/home/seal/1.txt");
			//forward convolution data
			for (int g = 0; g < _group; ++g)
			cacu_sgemm(NOTRANS, NOTRANS, col_data_->s_data() + col_offset * g, o_blob_->channel_length(),_w->length() / _group, _w->s_data() + w_offset * g, _w->num() / _group, (float_t)1, o_blob_->p_data(i) + out_offset * g,(float_t)0);
			//cacu_print(o_blob_->p_data(i),1000);
			//add bias
			if(_is_use_bias)
			cacu_sgemm(NOTRANS, NOTRANS, bias_multiplier->s_data(), bias_multiplier->count(), 1, _bias->s_data(), _bias->count(),(float_t)(1),o_blob_->p_data(i),(float_t)(1));
			//cacu_ssxpy(_bias->s_data(), (float_t)(1), _bias->count(), o_blob_->p_data(i), (float_t)(1), o_blob_->length(), o_blob_->p_data(i));
		}
#endif
	}

	virtual const void grad() override {

		col_offset = s_blob->channel() / _group * _col_data->width()
				* _col_data->height();
		w_offset = _w->count() / _group / _group;
		out_offset = _w->num() / _group * o_blob->width() * o_blob->height();

		blob *col_data_ = (blob*) _col_data;
		blob *bias_multiplier = (blob*) _bias_multiplier;

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;

		for (int i = 0; i < s_blob_->num(); ++i) {

			//gradient propagation
			for (int g = 0; g < _group; ++g)
				//cacu_sgemm(NOTRANS,TRANS, _w->s_data() + w_offset * g, _w->length() / _group, _w->num() / _group, o_blob_->p_diff(i) + out_offset * g, o_blob_->width() * o_blob_->height(), 1, col_data_->s_diff() + col_offset * g, 0);
				cacu_sgemm(NOTRANS, TRANS,
						o_blob_->p_diff_d(i) + out_offset * g,
						o_blob_->width() * o_blob_->height(),
						_w->num() / _group, _w->s_data() + w_offset * g,
						_w->length() / _group, 1,
						col_data_->s_diff() + col_offset * g, 0);
			//col2img
			//unpadded
			cacu_col2img_pad(col_data_->s_diff(), _args->kernel_size(),
					_args->stride(), s_blob->width(), s_blob->height(),
					s_blob->channel(), o_blob_->width(), o_blob_->height(),
					_args->pad(), _args->pad(), s_blob_->p_diff_d(i));
			//weights gradient
			cacu_img2col_pad(s_blob_->p_data_d(i), _args->kernel_size(),
					_args->stride(), s_blob_->width(), s_blob_->channel(),
					o_blob_->width(), _args->pad(), col_data_->s_data());
			for (int g = 0; g < _group; ++g)
				cacu_sgemm(TRANS, NOTRANS, col_data_->s_data() + col_offset * g,
						_w->length() / _group, o_blob_->channel_length(),
						o_blob_->p_diff_d(i) + out_offset * g,
						_w->num() / _group, 1, _w->s_diff() + w_offset * g, 1);
			//bias gradient
			if (_is_use_bias)
				//cacu_sumbysize(BYWIDTH,o_blob_->p_diff_d(i),o_blob_->length(),1,_bias->s_diff(),1,o_blob_->width()*o_blob_->height());
				cacu_sgemv(TRANS, o_blob_->p_diff_d(i),
						bias_multiplier->count(), bias_multiplier->s_data(),
						o_blob_->channel(), (float_t) (1), _bias->s_diff(),
						(float_t) (1));
			s_blob_->_sync(i);
		}
#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blob;

		for (int i = 0; i < s_blob_->num(); ++i) {
			//gradient propagation
			for (int g = 0; g < _group; ++g)
			//cacu_sgemm(NOTRANS,TRANS, _w->s_data() + w_offset * g, _w->length() / _group, _w->num() / _group, o_blob_->p_diff(i) + out_offset * g, o_blob_->width() * o_blob_->height(), 1, col_data_->s_diff() + col_offset * g, 0);
			cacu_sgemm(NOTRANS,TRANS, o_blob_->p_diff(i) + out_offset * g, o_blob_->width() * o_blob_->height(), _w->num() / _group, _w->s_data() + w_offset * g, _w->length() / _group, 1, col_data_->s_diff() + col_offset * g, 0);
			//col2img
			//unpadded
			cacu_col2img_pad(col_data_->s_diff(),_args->kernel_size(),_args->stride(),s_blob->width(),s_blob->height(),s_blob->channel(),o_blob_->width(),o_blob_->height(),_args->pad(),_args->pad(), s_blob_->p_diff(i));
			//weights gradient
			cacu_img2col_pad(s_blob_->p_data(i), _args->kernel_size(), _args->stride(),s_blob->width(),s_blob->height(),s_blob->channel(),o_blob_->width(),o_blob_->height(),_args->pad(),_args->pad(), col_data_->s_data());
			for (int g = 0; g < _group; ++g)
			cacu_sgemm(TRANS,NOTRANS,col_data_->s_data() + col_offset * g, _w->length() / _group, o_blob_->channel_length(), o_blob_->p_diff(i) + out_offset * g, _w->num() / _group, 1, _w->s_diff() + w_offset * g, 1);
			//bias gradient
			if(_is_use_bias)
			//cacu_sumbysize(BYWIDTH,o_blob_->p_diff(i),o_blob_->length(),1,_bias->s_diff(),1,o_blob_->width()*o_blob_->height());
			cacu_sgemv(TRANS,o_blob_->p_diff(i),bias_multiplier->count(),bias_multiplier->s_data(),o_blob_->channel(),(float_t)(1),_bias->s_diff(),(float_t)(1));
		}
#endif
	}

	virtual const void load(std::ifstream& is) override {
		if (_group != 1) {
			_w->load_group(is, _group);
		} else
			_w->load(is);
		if (_is_use_bias)
			_bias->load(is);
	}

	virtual const void save(std::ostream& os) override {
		if (_group != 1) {
			_w->serializa_group(os, _group);
		} else
			_w->serializa(os);
		if (_is_use_bias)
			_bias->serializa(os);
	}

	virtual const void echo() override
	{
		LOG_INFO("create convolution op:");
		LOG_INFO(
				"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d), kenrel_size: %d, stride: %d, pad: %d",
				s_blob->channel(), s_blob->width(), s_blob->height(),
				o_blob->channel(), o_blob->width(), o_blob->height(),
				_args->kernel_size(), _args->stride(), _args->pad());
	}

	inline virtual const void LOOP_INIT_DATA_() override
	{
		o_blob->_RESET_DATA();
		_w->_RESET_DIFF();
		if (_is_use_bias)
			_bias->_RESET_DIFF();
		_col_data->_RESET_DATA();
	}

	inline virtual const void set_phase(phase_type phase_) override {
		_phase = phase_;
	}

	inline void set_weight_init_type(param_init_type _type,
			float_t value = 0.0) {
		set_param_init_type(_type, _w, value);
	}

	inline void set_bias_init_type(param_init_type _type, float_t value = 0.0) {
		set_param_init_type(_type, _bias, value);
	}

	inline void set_group(int group) {
		CHECK_GT_OP(group, 0, "group must > 0 vs %d", group);
		CHECK_LE_OP(group, s_blob->channel(), "group must <= %d vs %d",
				_args->channel(), group);
		CHECK_EQ_OP(s_blob->channel() % group, 0,
				"channel mod group must == 0 vs %d", _args->channel() % group);
		this->_group = group;
	}

	void set_is_use_bias(bool switcher_) {
		_is_use_bias = switcher_;
	}
	;

protected:

	bool _is_use_bias = true;

	weight *_w;

	weight *_bias;

	blob *_col_data = NULL;

	blob *_bias_multiplier = NULL;

	int _group = 1;

private:

	int col_offset;

	int w_offset;

	int out_offset;

};
}
