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

#include "spatial_instance_normalize_op.h"
#include "../math/math.h"

#include "../../tools/vec_utils.h"

namespace cacu {

spatial_instance_normalize_op::spatial_instance_normalize_op(blobs *&data, data_args *&args_) :
		bn_base_op(data, args_, CACU_SPATIAL_INSTANCE_NORMALIZE) {

#if __USE_CUDNN__ == ON
	create_cudnn_handle(_in_handle);
	create_tensor_4d_desc(_data_desc);
	create_tensor_4d_desc(_mean_desc);
#endif

	_INIT_OP();

}

spatial_instance_normalize_op::~spatial_instance_normalize_op() {

#if __USE_CUDNN__ == ON
	release_cudnn_handle(_in_handle);
	destroy_tensor_descriptor(_data_desc);
	destroy_tensor_descriptor(_mean_desc);
#endif

}

void spatial_instance_normalize_op::initial() {

	if (o_blobs == NULL) {

		o_blobs = create_oblobs();
		o_blobs->push_back(
				create_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
						s_blobs->at(0)->width(), s_blobs->at(0)->height(),
						_phase));

		_mean = create_opblob(s_blobs->at(0)->num(), _args->output_channel(), 1,
				1, _phase);
		_var = create_opblob(s_blobs->at(0)->num(), _args->output_channel(), 1,
				1, _phase);

#if __USE_CUDNN__ == OFF
		//save for train
		_x = create_opblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
				s_blobs->at(0)->width(), s_blobs->at(0)->height(), test);
		_moving_scalar = create_opblob(1, 1, 1, 1, test);
		_one = create_opblob(1, 1, 1, 1, 1, test);
		_mutipler = create_opblob(1, s_blobs->at(0)->channel_length(), 1, 1,
				(float_t) 1.0, test);
		_mutipler->set_variable(false);

		_std = create_opblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), 1, 1, _phase);

#endif

		_scale = create_opblob(s_blobs->at(0)->num(), _args->output_channel(), 1, 1, _phase);
		_shift = create_opblob(s_blobs->at(0)->num(), _args->output_channel(), 1, 1, _phase);

	} else {
#if __USE_CUDNN__ == OFF
		o_blobs->at(0)->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
				s_blobs->at(0)->width(), s_blobs->at(0)->height());
		//save for train
		_x->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
				s_blobs->at(0)->width(), s_blobs->at(0)->height());

		_mutipler->resize(1, s_blobs->at(0)->channel_length(), 1, 1);
		_mutipler->set_data((float_t) 1.0);

		_std->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), 1, 1);

#endif
		_mean->resize(s_blobs->at(0)->num(), _args->output_channel(), 1, 1);
		_var->resize(s_blobs->at(0)->num(), _args->output_channel(), 1, 1);

		_scale->resize(s_blobs->at(0)->num(), _args->output_channel(), 1, 1);
		_shift->resize(s_blobs->at(0)->num(), _args->output_channel(), 1, 1);
	}

#if __USE_CUDNN__ == ON

	blob *o_blob_ = o_blobs->asblob(0);
	blob *s_blob_ = s_blobs->asblob(0);
	if (!s_blob_->body()->check_body(_NONE_BLOB)) {

		CHECK_EQ_OP(s_blob_->channel(), _scale->channel(),
				"Input data size is not matched! channel %d vs %d!",
				s_blob_->channel(), _scale->channel());

		CHECK_EQ_OP(s_blob_->channel(), _shift->channel(),
				"Input data size is not matched! channel %d vs %d!",
				s_blob_->channel(), _shift->channel());
	}
	set_tensor_4d_desc(_data_desc, 1, _args->output_channel(),
			s_blobs->at(0)->width(), s_blobs->at(0)->height());
	set_tensor_4d_desc(_mean_desc, 1, _args->output_channel(), 1, 1);

#endif

}

void spatial_instance_normalize_op::init_weights() {

	_scale_w = create_param("scale_w", _args->output_channel(), s_blobs->at(0)->channel_length(), 1, 1, _phase);
	_scale_w->set_init_type(xavier);
	_shift_w = create_param("shift_w", _args->output_channel(), s_blobs->at(0)->channel_length(), 1, 1, _phase);
	_shift_w->set_init_type(xavier);

}

void spatial_instance_normalize_op::check() {
	if (_args == NULL)
		LOG_FATAL("Spatical instance normalize data args cannot equal to NULL!");

}

void spatial_instance_normalize_op::op(blobs *&s_blobs_, blobs *&o_blobs_) {

	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);

	float_t m = (float_t) s_blob_->width() * s_blob_->height();

#if __USE_CUDNN__ == OFF

	blob *x_ = _x;

	cacu_sgemv(TRANS, s_blob_->s_data(), _mutipler->count(),
			_mutipler->s_data(), _mean->count(), (float_t) (1.0 / m),
			_mean->s_data(), 0);

	for (int i = 0; i < s_blob_->num(); ++i) {

		cacu_ssxpy(_mean->p_data(i), (float_t) (-1.), _mean->length(),
				s_blob_->p_data(i), (float_t) (1.), s_blob_->length(),
				o_blob_->p_data(i));

		//for saving space here we use x_ for container calculate x^2
		cacu_sqr(o_blob_->p_data(i), o_blob_->length(), x_->p_data(i));

		cacu_sgemv(TRANS, x_->p_data(i), _mutipler->count(), _mutipler->s_data(),
				_var->length(), (float_t) (1.0 / m), _var->p_data(i),
				(float_t) (0));

		cacu_stdbychannel(_var->p_data(i), _std->length(), _std->p_data(i),
				_epsilon);
		cacu_ssxpy(_mean->p_data(i), (float_t) (-1), _mean->length(),
				s_blob_->p_data(i), (float_t) (1), s_blob_->length(),
				o_blob_->p_data(i));
		cacu_cdxsize(o_blob_->p_data(i), s_blob_->length(), _std->p_data(i),
				_std->length(), o_blob_->p_data(i));
		//save for train
		cacu_copy(o_blob_->p_data(i), s_blob_->length(), x_->p_data(i));
		cacu_cxsize(o_blob_->p_data(i), s_blob_->length(), _scale->s_data(),
				_scale->count(), o_blob_->p_data(i));
		cacu_ssxpy(_shift->s_data(), (float_t) (1), _shift->count(),
				o_blob_->p_data(i), (float_t) (1), s_blob_->length(),
				o_blob_->p_data(i));
	}
#else

	for (int i = 0; i < s_blob_->num(); ++i) {

		cacu_sgemm(TRANS, NOTRANS, _scale_w->s_data(), _scale_w->num(), _scale_w->length(),
						s_blob_->p_data(i), s_blob_->channel(), (float_t) 1., _scale->p_data(i),
						(float_t) 0.);
		cacu_sgemm(TRANS, NOTRANS, _shift_w->s_data(), _shift_w->num(), _shift_w->length(),
						s_blob_->p_data(i), s_blob_->channel(), (float_t) 1., _shift->p_data(i),
						(float_t) 0.);

		cudnn_bn_training_forward(_in_handle, _data_desc, s_blob_->p_data(i),
				_bn_mode, _mean_desc, _scale->p_data(i), _shift->p_data(i), 0.0,
				_mean->p_data(i), _var->p_data(i), _epsilon, _mean->p_data(i),
				_var->p_data(i), _data_desc, o_blob_->p_data(i));
	}

#endif
}

void spatial_instance_normalize_op::grad(blobs *&s_blobs_, blobs *&o_blobs_) {

	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);

#if __USE_CUDNN__ == OFF
	float_t *mean_data_, *mean_diff_;

	blob *x_ = _x;

	for (int i = 0; i < s_blob_->num(); ++i) {
		//calculate dl/x_
		cacu_cxsize(o_blob_->p_diff(i), s_blob_->length(), _scale->s_data(),
				_scale->count(), s_blob_->p_diff(i));

		mean_data_ = _mean->p_data(i);
		mean_diff_ = _mean->p_diff(i);

		//calculate dl/std^2
		cacu_bn_rou_grad(s_blob_->p_data(i), s_blob_->p_diff(i), mean_data_,
				_std->p_data(i), 1, s_blob_->length(), s_blob_->channel(),
				_std->p_diff(i));
		//calculate dl/mu
		cacu_bn_mu_grad(s_blob_->p_data(i), s_blob_->p_diff(i), mean_data_,
				_std->p_data(i), _std->p_diff(i), 1, s_blob_->length(),
				s_blob_->channel(), mean_diff_);
		//calculate dl/x
		cacu_bn_dx_grad(s_blob_->p_data(i), s_blob_->p_diff(i), mean_data_,
				_std->p_data(i), _std->p_diff(i), mean_diff_, 1,
				s_blob_->length(), s_blob_->channel(), s_blob_->p_diff(i));
		//gradient of scale
		cacu_bn_gamma_grad(x_->p_data(i), o_blob_->p_diff(i), 1,
				s_blob_->length(), s_blob_->channel(), _scale->s_diff());
		//gradient of shift
		cacu_sgemv(TRANS, o_blob_->p_diff(i), _mutipler->count(),
				_mutipler->s_data(), _shift->count(), (float_t) (1),
				_shift->s_diff(), 1);
	}
#else

	for (int i = 0; i < s_blob_->num(); ++i) {
		cudnn_bn_backward(_in_handle, _data_desc, s_blob_->p_data(i), _bn_mode,
				_mean_desc, _data_desc, s_blob_->p_diff(i), _scale->p_data(i),
				_scale->p_diff(i), _shift->p_diff(i), _epsilon, NULL,
				NULL, _data_desc, o_blob_->p_data(i), _data_desc,
				o_blob_->p_diff(i));

		cacu_sgemm(NOTRANS, NOTRANS, _scale_w->s_data(), _scale_w->length(), _scale_w->num(),
				o_blob_->p_diff(i), o_blob_->channel(), (float_t) 1., s_blob_->p_diff(i),
				(float_t) 1.);
		cacu_sgemm(NOTRANS, NOTRANS, _shift_w->s_data(), _shift_w->length(), _shift_w->num(),
				o_blob_->p_diff(i), o_blob_->channel(), (float_t) 1., s_blob_->p_diff(i),
				(float_t) 1.);
		cacu_sgemm(NOTRANS, TRANS, s_blob_->p_data(i), s_blob_->channel_length(),
				s_blob_->channel(), _scale->p_diff(i), o_blob_->num(),
				(float_t) 1., _scale_w->s_diff(), (float_t) 1.);
		cacu_sgemm(NOTRANS, TRANS, s_blob_->p_data(i), s_blob_->channel_length(),
				s_blob_->channel(), _shift->p_diff(i), o_blob_->num(),
				(float_t) 1., _shift_w->s_diff(), (float_t) 1.);
	}

#endif

}

void spatial_instance_normalize_op::load(std::ifstream& is) {

	_scale_w->load(is);
	_shift_w->load(is);

}

void spatial_instance_normalize_op::save(std::ostream& os) {

	_scale_w->serializa(os);
	_shift_w->serializa(os);
}

void spatial_instance_normalize_op::echo() {
	LOG_INFO("Spatical instance normalize op: in(%d, %d, %d), out(%d, %d, %d)",
			s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
			s_blobs->at(0)->height(), o_blobs->at(0)->channel(),
			o_blobs->at(0)->width(), o_blobs->at(0)->height());
}

void spatial_instance_normalize_op::set_phase(phase_type phase_) {

	_phase = phase_;
}

void spatial_instance_normalize_op::set_scale_init_type(param_init_type _type,
		float_t mean, float_t std) {
	_scale->set_init_type(_type, std);
	cacu_sdxsize(_scale->s_data(), _scale->count(), mean, 1.0f,
			_scale->s_data());
}

void spatial_instance_normalize_op::update_history() {

}

REGISTER_OP_CLASS_D_DP(spatial_instance_normalize_op);
REGISTER_OP_MACRO(spatial_instance_normalize_op, CACU_SPATIAL_INSTANCE_NORMALIZE);

}
