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

#include "ip_projection_op.h"
#include "../math/math.h"

namespace cacu {

ip_projection_op::ip_projection_op(blobs *&data, op_args *&o_args_,
		data_args *&args_) :
		operator_base(data, o_args_, args_, CACU_IP_PROJECTION) {
	_INIT_OP();
}

ip_projection_op::~ip_projection_op() {

	delete _ws;
	_ws = NULL;
	delete _biases;
	_biases = NULL;
	if (_us != NULL)
		delete _us;
	if (_vs != NULL)
		delete _vs;
	if (_ws_original != NULL)
		delete _ws_original;
}

void ip_projection_op::initial() {
	if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
		o_blobs = create_em_oblobs();
		o_blobs = create_em_oblob(s_blobs->at(0)->num(), _args->output_channel(), 1,
				1, _phase);
#else
		o_blobs = create_oblobs();
		o_blobs->push_back(
				create_oblob(s_blobs->at(0)->num(), _args->output_channel(), 1,
						1, _phase));
#endif
		_bias_multiplier = create_opblob(1, 1, 1, 1, (float_t) (1), _phase);
		_bias_multiplier->set_variable(false);
	} else {
		o_blobs->at(0)->resize(s_blobs->at(0)->num(), _args->output_channel(),
				1, 1);
		_bias_multiplier->resize(1, 1, 1, 1);
		_bias_multiplier->set_data(1.0);
	}
}

void ip_projection_op::init_weights() {

	_ws = new vector<weight*>();
	_biases = new vector<weight*>();
	int size = (int) _o_args->at(0);
	for (int i = 0; i < size; ++i)
		_ws->push_back(
				create_param("w", _args->output_channel(), _args->channel(), 1,
						1, _phase));
	for (int i = 0; i < size; ++i)
		_biases->push_back(
				create_param("bias", _args->output_channel(), 1, 1, 1, _phase));
}

void ip_projection_op::check() {
	if (_args == NULL)
		LOG_FATAL("innerproduct projection data args cannot equal to NULL!");
	if (_o_args == NULL)
		LOG_FATAL("innerproduct projection op args cannot equal to NULL!");
	//output_channel > 0
	CHECK_GT_OP(_args->output_channel(), 0, "output_channel must > 0 vs %d",
			_args->output_channel());
}

void ip_projection_op::op(blobs *&s_blobs_, blobs *&o_blobs_) {

	if (_IS_NORM_WEIGHTS)
		norm_weight();

#if __USEMBEDDING__ == ON
	em_blob *o_blob_ = (em_blob*) o_blob;
	em_blob *s_blob_ = (em_blob*) s_blob;

	for (int i = 0; i < s_blob_->num(); ++i) {
		cacu_sgemm(TRANS, NOTRANS, _w->s_data(), _w->num(), _w->length(),
				s_blob_->p_data_d(i), 1, 1, o_blob_->p_data_d(i), 0);
		//bias added
		if (_is_use_bias)
		cacu_saxpby(_bias->s_data(), (float_t) (1),
				o_blob_->p_data_d(i), (float_t) (1), _bias->count());
		//cacu_ssxpy(_bias->s_data(),(float_t)(1),_bias->count(), o_blob_->p_data_d(i),(float_t)1,o_blob_->length(),o_blob_->p_data_d(i));
		o_blob_->_sync(i);
	}
#else
	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);
	bin_blob *label = s_blobs_->asbinblob(1);

	vec_i _target(label->count());
#if __USE_DEVICE__ == ON
	cuda_copy2host(&_target[0], label->s_data(), label->count());
#else
	for(int i = 0; i < s_blob_->num(); ++i)
	_target[i] = label->s_data()[i];
#endif
	for (int i = 0; i < s_blob_->num(); ++i) {
		weight *_w = _ws->at(_target[i]);
		weight *_bias = _biases->at(_target[i]);
		cacu_sgemm(TRANS, NOTRANS, _w->s_data(), _w->num(), _w->length(),
				s_blob_->p_data(i), 1, (float_t) 1., o_blob_->p_data(i),
				(float_t) 0);
		//bias added
		if (_is_use_bias)
			cacu_saxpby(_bias->s_data(), (float_t) (1), o_blob_->p_data(i),
					(float_t) (1), _bias->count());
		//cacu_ssxpy(_bias->s_data(),(float_t)(1),_bias->count(), o_blob_->p_data(i),(float_t)1,o_blob_->length(),o_blob_->p_data(i));
	}
#endif
}

void ip_projection_op::grad(blobs *&s_blobs_, blobs *&o_blobs_) {

	//blob *bias_multiplier = (blob*) _bias_multiplier;

#if __USEMBEDDING__ == ON
	em_blob *o_blob_ = (em_blob*) o_blob;
	em_blob *s_blob_ = (em_blob*) s_blob;

	for (int i = 0; i < s_blob_->num(); ++i) {
		//gradient propagation
		cacu_sgemm(NOTRANS, NOTRANS, _w->s_data(), _w->length(), _w->num(),
				o_blob_->p_diff_d(i), 1, 1, s_blob_->p_diff_d(i), 0);
		//weights gradient
		cacu_sgemm(NOTRANS, TRANS, s_blob_->p_data_d(i), s_blob_->length(),
				1, o_blob_->p_diff_d(i), o_blob_->length(), 1, _w->s_diff(),
				1);
		if (_is_use_bias)
		//bias gradient
		cacu_sumbysize(BYHEIGHT, o_blob_->p_diff_d(i),
				o_blob_->length(), 1, _bias->s_diff(), 1,
				_bias->count());
		s_blob_->_sync(i);
	}
#else
	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);
	bin_blob *label = s_blobs_->asbinblob(1);

	vec_i _target(label->count());
#if __USE_DEVICE__ == ON
	cuda_copy2host(&_target[0], label->s_data(), label->count());
#else
	for(int i = 0; i < s_blob_->num(); ++i)
	_target[i] = label->s_data()[i];
#endif
	for (int i = 0; i < _ws->size(); ++i) {
		_ws->at(i)->set_update(false);
		_biases->at(i)->set_update(false);
	}
	for (int i = 0; i < s_blob_->num(); ++i) {
		weight *_w = _ws->at(_target[i]);
		weight *_bias = _biases->at(_target[i]);
		_w->set_update(true);
		_bias->set_update(true);

		//gradient propagation
		cacu_sgemm(NOTRANS, NOTRANS, _w->s_data(), _w->length(), _w->num(),
				o_blob_->p_diff(i), 1, (float_t) 1., s_blob_->p_diff(i),
				(float_t) 0.);
		//weights gradient
		cacu_sgemm(NOTRANS, TRANS, s_blob_->p_data(i), s_blob_->length(), 1,
				o_blob_->p_diff(i), _args->output_channel(), (float_t) 1.,
				_w->s_diff(), (float_t) 1.);
		if (_is_use_bias)
			//bias gradient
			cacu_saxpby(o_blob_->p_diff(i), (float_t) 1, _bias->s_diff(),
					(float_t) (1), o_blob_->length());
		//cacu_sgemv(NOTRANS, o_blob_->p_diff(i), _args->output_channel(), bias_multiplier->s_data(), bias_multiplier->count(), (float_t)(1), _bias->s_diff(), (float_t)(1));
	}
	//cacu_sumbysize(BYHEIGHT,o_blob_->s_diff(),o_blob_->count(),1 ,_bias->s_diff(),1,_bias->count());
#endif
	if (_IS_NORM_WEIGHTS)
		norm_recover();
}

void ip_projection_op::load(std::ifstream& is) {
	for (int i = 0; i < _ws->size(); ++i)
		_ws->at(i)->load(is);
	if (_is_use_bias)
		for (int i = 0; i < _biases->size(); ++i)
			_biases->at(i)->load(is);
//		if(_us != NULL)
//			for (int i = 0 ; i < _us->size(); ++i)
//				_us->astype<blob>(i)->load(is);
}

void ip_projection_op::save(std::ostream& os) {
	for (int i = 0; i < _ws->size(); ++i)
		_ws->at(i)->serializa(os);
	if (_is_use_bias)
		for (int i = 0; i < _biases->size(); ++i)
			_biases->at(i)->serializa(os);
//		if(_us != NULL)
//			for (int i = 0 ; i < _us->size(); ++i)
//				_us->astype<blob>(i)->serializa(os);
}

void ip_projection_op::echo() {
	LOG_INFO("create innerproduct projection op:");
	LOG_INFO("innerproduct projection op: in(%d, %d, %d), out(%d, %d, %d), class(%d)",
			s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
			s_blobs->at(0)->height(), o_blobs->at(0)->channel(),
			o_blobs->at(0)->width(), o_blobs->at(0)->height(), (int) _o_args->at(0));
}

void ip_projection_op::norm_weight() {
	int size = (int) _o_args->at(0);
	if (_ws_original == NULL && _us == NULL && _vs == NULL) {
		LOG_INFO("Initalize the parameters for spectral normalization!");
		_us = new blobs();
		_vs = new blobs();
		_ws_original = new blobs();
		for (int i = 0; i < size; ++i) {
			weight *_w = _ws->at(i);
			_us->push_back(new blob(_w->num(), 1, 1, 1, 0, train));
			_vs->push_back(
					new blob(1, _w->channel(), _w->width(), _w->height(), 0,
							test));
			_ws_original->push_back(
					new blob(_w->num(), _w->channel(), _w->width(),
							_w->height(), 0, test));
			_us->asblob(i)->set_init_type(gaussian, 1);
		}
	}
	for (int j = 0; j < size; ++j) {
		weight *_w = _ws->at(j);
		blob *_w_orig = _ws_original->asblob(j);
		blob *_u = _us->asblob(j);
		blob *_v = _vs->asblob(j);
		cacu_copy(_w->s_data(), _w->count(), _w_orig->s_data());
		//_u->set_init_type(gaussian, 1);
		for (int i = 0; i < _pow_iter; ++i) {
			//cacu_sgemm(NOTRANS,TRANS,_u->s_data(),1,_u->count(),_w->s_data(),_w->length(),(float_t)1.,_v->s_data(),(float_t)0);
			cacu_sgemv(NOTRANS, _w->s_data(), _w->length(), _u->s_data(),
					_w->num(), (float_t) 1., _v->s_data(), (float_t) 0);
			cacu_l2_normalization(_v->s_data(), _v->count(), 1e-12);

			//cacu_sgemm(NOTRANS,NOTRANS,_v->s_data(),1,_v->count(),_w->s_data(),_w->num(),(float_t)1.,_u->s_data(),(float_t)0);
			cacu_sgemv(TRANS, _w->s_data(), _w->length(), _v->s_data(),
					_w->num(), (float_t) 1., _u->s_data(), (float_t) 0);
			cacu_l2_normalization(_u->s_data(), _u->count(), 1e-12);
		}
		//cacu_sgemm(NOTRANS,NOTRANS,_v->s_data(),1,_v->count(),_w->s_data(),_w->num(),(float_t)1.,_u->s_diff(),(float_t)0);
		cacu_sgemv(TRANS, _w->s_data(), _w->length(), _v->s_data(), _w->num(),
				(float_t) 1., _u->s_diff(), (float_t) 0);
		//cacu_sgemm(NOTRANS,NOTRANS,_u->s_diff(),1,_u->count(),_u->s_data(),1, (float_t)1.,_v->s_data(),(float_t)0);
		cacu_sgemv(NOTRANS, _u->s_diff(), 1, _u->s_data(), _w->num(),
				(float_t) 1., _v->s_data(), (float_t) 0);
		//_ws->switch_channel();

		float_t _sigma = 0.;
#if __USE_DEVICE__
		device_copy2host(&_sigma, _v->s_data(), 1);
#else
		_sigma = _v->s_data()[0];
#endif
		cacu_scalex(_w->s_data(), _w->count(), (float_t) 1. / _sigma);
	}
}

void ip_projection_op::norm_recover() {
	for (int j = 0; j < _o_args->at(0); ++j) {
		weight *_w = _ws->at(j);
		blob *_w_orig = _ws_original->asblob(j);
		cacu_copy(_w_orig->s_data(), _w->count(), _w->s_data());
	}
}

REGISTER_OP_CLASS_D_ODP(ip_projection_op);
REGISTER_OP_MACRO(ip_projection_op, CACU_IP_PROJECTION);

}

