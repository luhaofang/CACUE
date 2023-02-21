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

#include "blur_op.h"
#include "../math/math.h"

#include "../../tools/vec_utils.h"

namespace cacu {

//output_channel, kernel_size, stride, pad, input_dim, channel
blur_op::blur_op(blobs *&data, data_args *&args_) :
		conv_base_op(data, args_, CACU_BLUR) {
	_INIT_OP();
}

blur_op::~blur_op() {

}

void blur_op::initial() {

	int output_w = s_blobs->at(0)->width();
	int output_h = s_blobs->at(0)->height();

	if (o_blobs == NULL) {
		o_blobs = s_blobs;
#if __USE_CUDNN__ == OFF
		_col_data = create_opblob(1, s_blobs->at(0)->channel(),
				output_w * 3, output_h * 3, _phase);
#endif

	} else {

		o_blobs->_NEED_MOTIFY();

#if __USE_CUDNN__ == OFF
		_col_data->resize(1, s_blobs->at(0)->channel(),
				output_w * 3, output_h * 3);
#endif
	}

	this->set_group(s_blobs->at(0)->channel());
#if __USE_CUDNN__ == ON

	blob *s_blob_ = s_blobs->asblob(0);

	if (!s_blob_->body()->check_body(_NONE_BLOB)) {

		CHECK_EQ_OP(s_blob_->channel(), _w->channel(),
				"Input data size is not matched! channel %d vs %d!",
				s_blob_->channel(), _w->channel());

		set_convolution_desc(_conv_desc, 1, 1, 1, 1);
		set_tensor_4d_desc(_s_blob_desc, s_blob_->num(),
				s_blob_->channel() / _group, s_blob_->width(),
				s_blob_->height());

		get_convolution_fwd_algo(_conv_handle, _fwd_algo, _s_blob_desc,
				_w->filter_desc(), _conv_desc, _o_blob_desc);
		get_convolution_fwd_work_space(_conv_handle, _fwd_workspace,
				_s_blob_desc, _w->filter_desc(), _conv_desc, _o_blob_desc,
				_fwd_algo);
		if (_workspace_fwd == NULL)
			_workspace_fwd = create_opblob(1, _fwd_workspace, 1, 1, _phase);
		else
			_workspace_fwd->resize(1, _fwd_workspace, 1, 1);

		if (_phase == train) {

			get_convolution_bwd_filter_algo(_conv_handle, _bwd_w_algo,
					_s_blob_desc, _w->filter_desc(), _conv_desc, _o_blob_desc);
			get_convolution_bwd_filter_work_space(_conv_handle,
					_bwd_w_workspace, _s_blob_desc, _w->filter_desc(),
					_conv_desc, _o_blob_desc, _bwd_w_algo);
			if (_workspace_bwd_w == NULL)
				_workspace_bwd_w = create_opblob(1, _bwd_w_workspace, 1, 1,
						_phase);
			else
				_workspace_bwd_w->resize(1, _bwd_w_workspace, 1, 1);

			get_convolution_bwd_algo(_conv_handle, _bwd_algo, _s_blob_desc,
					_w->filter_desc(), _conv_desc, _o_blob_desc);
			get_convolution_bwd_work_space(_conv_handle, _bwd_workspace,
					_s_blob_desc, _w->filter_desc(), _conv_desc, _o_blob_desc,
					_bwd_algo);
			if (_workspace_bwd == NULL)
				_workspace_bwd = create_opblob(1, _bwd_workspace, 1, 1, _phase);
			else
				_workspace_bwd->resize(1, _bwd_workspace, 1, 1);
		}
	}
#endif
}

void blur_op::init_weights() {

	_w = create_param("w", _args->channel(), _args->channel(), 3, 3, _phase);
	vec_t w_ = {0.0625, 0.1250, 0.0625, 0.1250, 0.2500, 0.1250, 0.0625, 0.1250, 0.0625};
	LOG_DEBUG("%d ", w_.size());

	for(int n = 0; n < _args->channel(); ++n)
		for(int c = 0; c < _args->channel(); ++c)
			_w->copy2data(w_, _w->length() * n + c * _w->channel_length(), _w->channel_length());
	_w->set_variable(false);
	_w->set_update(false);
}

void blur_op::op(blobs *&s_blobs_, blobs *&o_blobs_) {

	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);

	w_offset = _w->count() / _group / _group;

#if __USE_CUDNN__ == OFF

	col_offset = s_blobs->at(0)->channel() / _group * _col_data->channel_length();
	out_offset = _w->num() / _group * o_blobs->at(0)->channel_length();

	blob *col_data_ = _col_data;

	for (int i = 0; i < s_blob_->num(); ++i) {
		//padded data if needed & img2col change
		cacu_img2col_pad(s_blob_->p_data(i), 3,
				3, 1, s_blob_->width(),
				s_blob_->height(), s_blob_->channel(), o_blob_->width(),
				o_blob_->height(), 1, 1,
				col_data_->s_data());
		//forward convolution data
		for (int g = 0; g < _group; ++g)
		cacu_sgemm(NOTRANS, NOTRANS, col_data_->s_data() + col_offset * g,
				o_blob_->channel_length(), _w->length() / _group,
				_w->s_data() + w_offset * g, _w->num() / _group,
				(float_t) 1, o_blob_->p_data(i) + out_offset * g,
				(float_t) 0);

	}
#else

	if (_group != 1)
		s_blob_->switch_channel_data(s_blob_->num(), _group,
				s_blob_->channel() / _group * s_blob_->channel_length());

	for (int g = 0; g < _group; ++g) {
		cudnn_convolution_forward_data(_conv_handle, _s_blob_desc,
				s_blob_->s_data() + s_blob_->count() / _group * g,
				_w->filter_desc(), _w->s_data() + w_offset * g, _conv_desc,
				_o_blob_desc, o_blob_->s_data() + o_blob_->count() / _group * g,
				_fwd_algo, _workspace_fwd->s_data(), _fwd_workspace);
	}

	if (_group != 1)
		o_blob_->switch_channel_data(_group, o_blob_->num(),
				o_blob_->channel() / _group * o_blob_->channel_length());

#endif

}

void blur_op::grad(blobs *&s_blobs_, blobs *&o_blobs_) {

	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);

#if __USE_CUDNN__ == OFF

	blob *col_data_ = _col_data;

	col_offset = s_blobs->at(0)->channel() / _group * _col_data->channel_length();

	for (int i = 0; i < s_blob_->num(); ++i) {
		//if(_NEED_BACK_PROPAGATE_FEATURE)
		{
			//gradient propagation
			for (int g = 0; g < _group; ++g) {
				cacu_sgemm(NOTRANS, TRANS, o_blob_->p_diff(i) + out_offset * g,
						o_blob_->channel_length(), o_blob_->channel() / _group,
						_w->s_data() + w_offset * g, _w->length() / _group,
						(float_t) 1., col_data_->s_diff() + col_offset * g,
						(float_t) 0.);
			}
			//col2img
			//unpadded
			cacu_col2img_pad(col_data_->s_diff(), 3,
					3, 1, s_blob_->width(),
					s_blob_->height(), s_blob_->channel(), o_blob_->width(),
					o_blob_->height(), 1, 1,
					s_blob_->p_diff(i));
		}
	}
#else

	if (_group != 1)
		o_blob_->switch_channel_data(o_blob_->num(), _group,
				o_blob_->channel() / _group * o_blob_->channel_length());

	for (int g = 0; g < _group; ++g) {
		cudnn_convolution_backward_data(_conv_handle, _s_blob_desc,
				s_blob_->s_diff() + s_blob_->count() / _group * g,
				_w->filter_desc(), _w->s_data() + w_offset * g, _conv_desc,
				_o_blob_desc, o_blob_->s_diff() + o_blob_->count() / _group * g,
				_bwd_algo, _workspace_bwd->s_data(), _bwd_workspace);
	}

	if (_group != 1)
		s_blob_->switch_channel_data(s_blob_->num(), _group,
				s_blob_->channel() / _group * s_blob_->channel_length());

#endif

}

void blur_op::echo() {
	LOG_INFO(
			"blur op: in(%d, %d, %d), out(%d, %d, %d), kernel/stride/pad([%d, %d], %d, %d)",
			s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
			s_blobs->at(0)->height(), o_blobs->at(0)->channel(),
			o_blobs->at(0)->width(), o_blobs->at(0)->height(),
			_args->kernel_size(), _args->kernel_size(), _args->stride(),
			_args->pad());
}

void blur_op::load(std::ifstream& is) {
	if (_group != 1)
		_w->load_group(is, _group);
	else
		_w->load(is);
	if (_is_use_bias)
		_bias->load(is);
}

void blur_op::save(std::ostream& os) {
	if (_group != 1)
		_w->serializa_group(os, _group);
	else
		_w->serializa(os);
	if (_is_use_bias)
		_bias->serializa(os);
}

REGISTER_OP_CLASS_D_DP(blur_op);
REGISTER_OP_MACRO(blur_op, CACU_BLUR);

}
