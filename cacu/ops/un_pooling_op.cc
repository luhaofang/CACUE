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

#include "un_pooling_op.h"
#include "../math/math.h"

namespace cacu {

//output_channel, kernel_size, stride, pad, input_dim, channel
un_pooling_op::un_pooling_op(blobs *&data, data_args *&args_) :
		conv_base_op(data, args_, CACU_UN_POOLING) {
#if __USE_CUDNN__ == ON
	create_filter_desc(_filter_desc);
#endif
	_INIT_OP();
}

un_pooling_op::~un_pooling_op() {
#if __USE_CUDNN__ == ON
	destroy_filter_descriptor(_filter_desc);
#endif
}

void un_pooling_op::initial() {
	int input_w = s_blobs->at(0)->width();
	int input_h = s_blobs->at(0)->height();
	int num = s_blobs->at(0)->num();

	int output_w, output_h;
	if (input_w != 1) {
		output_w = input_w * _args->stride();
		_pad_w = ((input_w - 1) * _args->stride() + _args->kernel_size()
				- output_w);
		if (_pad_w % 2 != 0)
			_pad_w = _pad_w / 2 + 1;
		else
			_pad_w = _pad_w / 2;
	} else {
		output_w = (input_w - 1) * _args->stride() + _args->kernel_size()
				- _args->pad() * 2;
		if (_args->kernel_size() == 1)
			output_w = input_w * _args->stride() - 2 * _args->pad();
		_pad_w = _args->pad();
	}
	if (input_h != 1) {
		output_h = input_h * _args->stride();
		_pad_h = ((input_h - 1) * _args->stride() + _args->kernel_size()
				- output_h);
		if (_pad_h % 2 != 0)
			_pad_h = _pad_h / 2 + 1;
		else
			_pad_h = _pad_h / 2;
	} else {
		output_h = (input_h - 1) * _args->stride() + _args->kernel_size()
				- _args->pad() * 2;
		if (_args->kernel_size() == 1)
			output_h = input_h * _args->stride() - 2 * _args->pad();
		_pad_h = _args->pad();
	}

	_group = _args->output_channel();

	if (o_blobs == NULL) {
		o_blobs = create_oblobs();
		o_blobs->push_back(
				create_oblob(num, _args->output_channel(), output_w, output_h,
						_phase));
#if __USE_CUDNN__ == OFF
		_col_data = create_opblob(1, _args->output_channel(),
				input_w * _args->kernel_size(), input_h * _args->kernel_size(),
				_phase);
		_bias_multiplier = create_opblob(1, 1, output_w, output_h, 1.0, _phase);
		_bias_multiplier->set_variable(false);
#endif
	} else {
		o_blobs->at(0)->resize(num, _args->output_channel(), output_w,
				output_h);
#if __USE_CUDNN__ == OFF
		_col_data->resize(1, _args->output_channel(),
				input_w * _args->kernel_size(), input_h * _args->kernel_size());
		_bias_multiplier->resize(1, 1, output_w, output_h);
		_bias_multiplier->set_data(1.0);
#endif
	}

#if __USE_CUDNN__ == ON
	blob *o_blob_ = o_blobs->asblob(0);
	blob *s_blob_ = s_blobs->asblob(0);

	if (!s_blob_->body()->check_body(_NONE_BLOB)) {

		CHECK_EQ_OP(s_blob_->channel(), _ws->num(),
				"Input data size is not matched! channel %d vs %d!",
				s_blob_->channel(), _ws->num());

		CHECK_EQ_OP(o_blob_->channel(), _ws->channel(),
				"Output data size is not matched! channel %d vs %d!",
				o_blob_->channel(), _ws->channel());

		set_convolution_desc(_conv_desc, _pad_w, _pad_h, _args->stride(),
				_args->stride());

		set_tensor_4d_desc(_s_blob_desc, s_blob_->num(),
				s_blob_->channel() / _group, s_blob_->width(),
				s_blob_->height());

		set_tensor_4d_desc(_o_blob_desc, o_blob_->num(),
				o_blob_->channel() / _group, o_blob_->width(),
				o_blob_->height());

		get_convolution_bwd_algo(_conv_handle, _bwd_algo, _o_blob_desc,
				_filter_desc, _conv_desc, _s_blob_desc);
		get_convolution_bwd_work_space(_conv_handle, _bwd_workspace,
				_o_blob_desc, _filter_desc, _conv_desc, _s_blob_desc,
				_bwd_algo);
		if (_workspace_bwd == NULL)
			_workspace_bwd = create_opblob(1, _bwd_workspace, 1, 1, _phase);
		else
			_workspace_bwd->resize(1, _bwd_workspace, 1, 1);

		if (_phase == train) {

			get_convolution_bwd_filter_algo(_conv_handle, _bwd_w_algo,
					_o_blob_desc, _filter_desc, _conv_desc, _s_blob_desc);
			get_convolution_bwd_filter_work_space(_conv_handle,
					_bwd_w_workspace, _o_blob_desc, _filter_desc, _conv_desc,
					_s_blob_desc, _bwd_w_algo);
			if (_workspace_bwd_w == NULL)
				_workspace_bwd_w = create_opblob(1, _bwd_w_workspace, 1, 1,
						_phase);
			else
				_workspace_bwd_w->resize(1, _bwd_w_workspace, 1, 1);

			get_convolution_fwd_algo(_conv_handle, _fwd_algo, _o_blob_desc,
					_filter_desc, _conv_desc, _s_blob_desc);
			get_convolution_fwd_work_space(_conv_handle, _fwd_workspace,
					_o_blob_desc, _filter_desc, _conv_desc, _s_blob_desc,
					_fwd_algo);
			if (_workspace_fwd == NULL)
				_workspace_fwd = create_opblob(1, _fwd_workspace, 1, 1, _phase);
			else
				_workspace_fwd->resize(1, _fwd_workspace, 1, 1);
		}
	}
#endif
}

void un_pooling_op::init_weights() {
	_ws = create_opblob(_args->channel(), _args->output_channel(),
			_args->kernel_size(), _args->kernel_size(), test);
	_ws->set_data((float_t) 1.);
	_ws->set_variable(false);

#if __USE_CUDNN__ == ON
	set_filter_4d_desc(_filter_desc, 1, 1, _args->kernel_size(),
			_args->kernel_size());
#endif
}

void un_pooling_op::check() {
	if (_args == NULL)
		LOG_FATAL("convolution data args cannot equal to NULL!");
	//output_channel < channel
	CHECK_GT_OP(_args->output_channel(), 0, "output_channel must > 0 vs %d",
			_args->output_channel());
	//kernel_size > 0
	CHECK_GT_OP(_args->kernel_size(), 0, "kernel_size must > 0 vs %d",
			_args->kernel_size());
	//stride > 0
	CHECK_GT_OP(_args->stride(), 0, "stride must > 0 vs %d", _args->stride());
	//in and out channel must be equal
	CHECK_EQ_OP(_args->channel(), _args->output_channel(),
			"channel and output_channel must be equal %d vs %d",
			_args->channel(), _args->output_channel());
	if (_args->size() > 5)
		_group = _args->at(6);
}

void un_pooling_op::op(blobs *&s_blobs_, blobs *&o_blobs_) {

#if __USE_CUDNN__ == ON

	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);

	w_offset = _ws->count() / _group / _group;

	if (_group != 1)
		s_blob_->switch_channel_data(s_blob_->num(), _group,
				s_blob_->channel() / _group * s_blob_->channel_length());

	for (int g = 0; g < _group; ++g) {
		cudnn_convolution_backward_data(_conv_handle, _o_blob_desc,
				o_blob_->s_data() + o_blob_->count() / _group * g, _filter_desc,
				_ws->s_data() + w_offset * g, _conv_desc, _s_blob_desc,
				s_blob_->s_data() + s_blob_->count() / _group * g, _bwd_algo,
				_workspace_bwd->s_data(), _bwd_workspace);
	}
	if (_group != 1)
		o_blob_->switch_channel_data(_group, o_blob_->num(),
				o_blob_->channel() / _group * o_blob_->channel_length());

#else
	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);
	blob *col_data_ = _col_data;

	col_offset = o_blob_->channel() / _group * _col_data->channel_length();
	w_offset = _ws->count() / _group / _group;
	out_offset = _ws->num() / _group * s_blob_->channel_length();

	for (int i = 0; i < s_blob_->num(); ++i) {
		//col_data_->blob_size();
		//gradient propagation
		for (int g = 0; g < _group; ++g)
		cacu_sgemm(NOTRANS, TRANS, s_blob_->p_data(i) + out_offset * g,
				s_blob_->width() * s_blob_->height(), _ws->num() / _group,
				_ws->s_data() + w_offset * g, _ws->length() / _group,
				(float_t) 1., col_data_->s_data() + col_offset * g,
				(float_t) 0);
		//col2img
		//unpadded
		cacu_col2img_pad(col_data_->s_data(), _args->kernel_size(),
				_args->kernel_size(), _args->stride(), o_blob_->width(),
				o_blob_->height(), o_blob_->channel(), s_blob_->width(),
				s_blob_->height(), _pad_w, _pad_h, o_blob_->p_data(i));
	}
#endif
}

void un_pooling_op::grad(blobs *&s_blobs_, blobs *&o_blobs_) {

#if __USE_CUDNN__ == ON
	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);

	w_offset = _ws->count() / _group / _group;
	out_offset = _ws->num() / _group * s_blob_->channel_length();

	if (_group != 1)
		o_blob_->switch_channel_data(o_blob_->num(), _group,
				o_blob_->channel() / _group * o_blob_->channel_length());

	for (int g = 0; g < _group; ++g) {
		cudnn_convolution_forward_data(_conv_handle, _o_blob_desc,
				o_blob_->s_diff() + o_blob_->count() / _group * g, _filter_desc,
				_ws->s_data() + w_offset * g, _conv_desc, _s_blob_desc,
				s_blob_->s_diff() + s_blob_->count() / _group * g, _fwd_algo,
				_workspace_fwd->s_data(), _fwd_workspace);
	}
	if (_group != 1)
		s_blob_->switch_channel_data(_group, s_blob_->num(),
				s_blob_->channel() / _group * s_blob_->channel_length());

#else
	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);
	blob *col_data_ = _col_data;

	col_offset = o_blob_->channel() / _group * _col_data->channel_length();
	w_offset = _ws->count() / _group / _group;
	out_offset = _ws->num() / _group * s_blob_->channel_length();

	for (int i = 0; i < s_blob_->num(); ++i) {
		//padded data if needed & img2col change
		cacu_img2col_pad(o_blob_->p_diff(i), _args->kernel_size(),
				_args->kernel_size(), _args->stride(), o_blob_->width(),
				o_blob_->height(), o_blob_->channel(), s_blob_->width(),
				s_blob_->height(), _pad_w, _pad_h, col_data_->s_diff());

		//forward convolution data
		for (int g = 0; g < _group; ++g) {
			cacu_sgemm(NOTRANS, NOTRANS, col_data_->s_diff() + col_offset * g,
					s_blob_->channel_length(), _ws->length() / _group,
					_ws->s_data() + w_offset * g, _ws->num() / _group,
					(float_t) 1, s_blob_->p_diff(i) + out_offset * g,
					(float_t) 0);
		}
	}
#endif
}

void un_pooling_op::echo() {
	LOG_INFO(
			"create un_pooling op: channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d), kenrel_size: %d, stride: %d, pad: %d",
			s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
			s_blobs->at(0)->height(), o_blobs->at(0)->channel(),
			o_blobs->at(0)->width(), o_blobs->at(0)->height(),
			_args->kernel_size(), _args->stride(), _args->pad());
}

REGISTER_OP_CLASS_D_DP(un_pooling_op);
REGISTER_OP_MACRO(un_pooling_op, CACU_UN_POOLING);

}
