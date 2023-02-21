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

#include "convolution_op.h"
#include "../math/math.h"

#include "../../tools/vec_utils.h"

namespace cacu {

//output_channel, kernel_size, stride, pad, input_dim, channel
convolution_op::convolution_op(blobs *&data, data_args *&args_) :
		conv_base_op(data, args_, CACU_CONVOLUTION) {
	_INIT_OP();
}

convolution_op::~convolution_op() {

}

void convolution_op::initial() {

	int output_w = (s_blobs->at(0)->width() + 2 * _args->pad()
			- _args->kernel_size()) / _args->stride() + 1;
	if (_args->kernel_size() == 1)
		output_w = (s_blobs->at(0)->width() + 2 * _args->pad())
				/ _args->stride();

	int output_h = (s_blobs->at(0)->height() + 2 * _args->pad()
			- _args->kernel_size()) / _args->stride() + 1;
	if (_args->kernel_size() == 1)
		output_h = (s_blobs->at(0)->height() + 2 * _args->pad())
				/ _args->stride();

	if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
		o_blobs = create_em_oblobs();
		o_blob = create_em_oblob(s_blob->num(), _args->output_channel(),
				output_w, output_h, _phase);

#else
		o_blobs = create_oblobs();
		o_blobs->push_back(
				create_oblob(s_blobs->at(0)->num(), _args->output_channel(),
						output_w, output_h, _phase));
#endif

#if __USE_CUDNN__ == OFF
		_col_data = create_opblob(1, s_blobs->at(0)->channel(),
				output_w * _args->kernel_size(),
				output_h * _args->kernel_size(), _phase);
		_bias_multiplier = create_opblob(1, 1, output_w, output_h,
				(float_t) (1), _phase);
		_bias_multiplier->set_variable(false);
#endif

	} else {

		o_blobs->at(0)->resize(s_blobs->at(0)->num(), _args->output_channel(),
				output_w, output_h);

#if __USE_CUDNN__ == OFF
		_col_data->resize(1, s_blobs->at(0)->channel(),
				output_w * _args->kernel_size(),
				output_h * _args->kernel_size());
		_bias_multiplier->resize(1, 1, output_w, output_h);
		_bias_multiplier->set_data(1.0);
#endif
	}
#if __USE_CUDNN__ == ON

	blob *o_blob_ = o_blobs->asblob(0);
	blob *s_blob_ = s_blobs->asblob(0);

	if (!s_blob_->body()->check_body(_NONE_BLOB)) {

		CHECK_EQ_OP(s_blob_->channel(), _w->channel(),
				"Input data size is not matched! channel %d vs %d!",
				s_blob_->channel(), _w->channel());

		CHECK_EQ_OP(o_blob_->channel(), _w->num(),
				"Output data size is not matched! channel %d vs %d!",
				o_blob_->channel(), _w->num());

		set_convolution_desc(_conv_desc, _args->pad(), _args->pad(),
				_args->stride(), _args->stride());
		set_tensor_4d_desc(_s_blob_desc, s_blob_->num(),
				s_blob_->channel() / _group, s_blob_->width(),
				s_blob_->height());
		set_tensor_4d_desc(_o_blob_desc, o_blob_->num(),
				o_blob_->channel() / _group, o_blob_->width(),
				o_blob_->height());

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

void convolution_op::init_weights() {

	_w = create_param("w", _args->output_channel(), _args->channel(),
			_args->kernel_size(), _args->kernel_size(), _phase);

	_bias = create_param("bias", _args->output_channel(), 1, 1, 1, _phase);

#if __USE_CUDNN__ == ON
	_bias->set_tensor_desc(1, _bias->num(), 1, 1);
#endif
}

void convolution_op::op(blobs *&s_blobs_, blobs *&o_blobs_) {

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
	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);

	w_offset = _w->count() / _group / _group;

#if __USE_CUDNN__ == OFF

	col_offset = s_blobs->at(0)->channel() / _group * _col_data->channel_length();
	out_offset = _w->num() / _group * o_blobs->at(0)->channel_length();

	blob *col_data_ = _col_data;
	blob *bias_multiplier = _bias_multiplier;

	for (int i = 0; i < s_blob_->num(); ++i) {
		//padded data if needed & img2col change
		cacu_img2col_pad(s_blob_->p_data(i), _args->kernel_size(),
				_args->kernel_size(), _args->stride(), s_blob_->width(),
				s_blob_->height(), s_blob_->channel(), o_blob_->width(),
				o_blob_->height(), _args->pad(), _args->pad(),
				col_data_->s_data());
		//forward convolution data
		for (int g = 0; g < _group; ++g)
		cacu_sgemm(NOTRANS, NOTRANS, col_data_->s_data() + col_offset * g,
				o_blob_->channel_length(), _w->length() / _group,
				_w->s_data() + w_offset * g, _w->num() / _group,
				(float_t) 1, o_blob_->p_data(i) + out_offset * g,
				(float_t) 0);
		//add bias
		if (_is_use_bias)
		cacu_sgemm(NOTRANS, NOTRANS, bias_multiplier->s_data(),
				bias_multiplier->count(), 1, _bias->s_data(),
				_bias->count(), (float_t) (1), o_blob_->p_data(i),
				(float_t) (1));

//			for(int c = 0; c < _w->num(); ++c){
//				if(FIND_FROM_VEC(*_w->upgrade_index(), c))
//					cacu_set_value(o_blob_->p_data(i) + c * o_blob_->channel_length(), 0, o_blob_->channel_length());
//			}
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
		if (_is_use_bias)
			cudnn_convolution_forward_bias(_conv_handle, _bias->tensor_desc(),
					_bias->s_data() + _bias->num() / _group * g, _o_blob_desc,
					o_blob_->s_data() + o_blob_->count() / _group * g);
	}

	if (_group != 1)
		o_blob_->switch_channel_data(_group, o_blob_->num(),
				o_blob_->channel() / _group * o_blob_->channel_length());

#endif

#endif
}

void convolution_op::grad(blobs *&s_blobs_, blobs *&o_blobs_) {

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
	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);

#if __USE_CUDNN__ == OFF

	blob *col_data_ = _col_data;
	blob *bias_multiplier = _bias_multiplier;

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
			cacu_col2img_pad(col_data_->s_diff(), _args->kernel_size(),
					_args->kernel_size(), _args->stride(), s_blob_->width(),
					s_blob_->height(), s_blob_->channel(), o_blob_->width(),
					o_blob_->height(), _args->pad(), _args->pad(),
					s_blob_->p_diff(i));
		}
		//weights gradient
		cacu_img2col_pad(s_blob_->p_data(i), _args->kernel_size(),
				_args->kernel_size(), _args->stride(), s_blob_->width(),
				s_blob_->height(), s_blob_->channel(), o_blob_->width(),
				o_blob_->height(), _args->pad(), _args->pad(),
				col_data_->s_data());
		for (int g = 0; g < _group; ++g)
		cacu_sgemm(TRANS, NOTRANS, col_data_->s_data() + col_offset * g,
				_w->length() / _group, o_blob_->channel_length(),
				o_blob_->p_diff(i) + out_offset * g,
				o_blob_->channel() / _group, (float_t) 1.,
				_w->s_diff() + w_offset * g, (float_t) 1.);
		//bias gradient
		if (_is_use_bias)
		cacu_sgemv(TRANS, o_blob_->p_diff(i), bias_multiplier->count(),
				bias_multiplier->s_data(), o_blob_->channel(),
				(float_t) (1), _bias->s_diff(), (float_t) (1));
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
		cudnn_convolution_backward_weight(_conv_handle, _s_blob_desc,
				s_blob_->s_data() + s_blob_->count() / _group * g,
				_w->filter_desc(), _w->s_diff() + w_offset * g, _conv_desc,
				_o_blob_desc, o_blob_->s_diff() + o_blob_->count() / _group * g,
				_bwd_w_algo, _workspace_bwd_w->s_data(), _bwd_w_workspace);
		if (_is_use_bias)
			cudnn_convolution_backward_bias(_conv_handle, _bias->tensor_desc(),
					_bias->s_diff() + _bias->num() / _group * g, _o_blob_desc,
					o_blob_->s_diff() + o_blob_->count() / _group * g);
	}

	if (_group != 1)
		s_blob_->switch_channel_data(s_blob_->num(), _group,
				s_blob_->channel() / _group * s_blob_->channel_length());

#endif
#endif
}

void convolution_op::echo() {
	LOG_INFO(
			"convolution op: in(%d, %d, %d), out(%d, %d, %d), kernel/stride/pad([%d, %d], %d, %d)",
			s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
			s_blobs->at(0)->height(), o_blobs->at(0)->channel(),
			o_blobs->at(0)->width(), o_blobs->at(0)->height(),
			_args->kernel_size(), _args->kernel_size(), _args->stride(),
			_args->pad());
}

void convolution_op::load(std::ifstream& is) {
	if (_group != 1)
		_w->load_group(is, _group);
	else
		_w->load(is);
	if (_is_use_bias)
		_bias->load(is);
}

void convolution_op::save(std::ostream& os) {
	if (_group != 1)
		_w->serializa_group(os, _group);
	else
		_w->serializa(os);
	if (_is_use_bias)
		_bias->serializa(os);
}

REGISTER_OP_CLASS_D_DP(convolution_op);
REGISTER_OP_MACRO(convolution_op, CACU_CONVOLUTION);

}
