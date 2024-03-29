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

#include "equalized_inner_product_op.h"
#include "../math/math.h"

#include "../../tools/vec_utils.h"

namespace cacu {

equalized_inner_product_op::equalized_inner_product_op(blobs *&data, data_args *&args_) :
		operator_base(data, args_, CACU_E_INNERPRODUCT) {
	_INIT_OP();
}

equalized_inner_product_op::~equalized_inner_product_op() {

}

void equalized_inner_product_op::initial() {
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
		_bias_multiplier = create_opblob(1, s_blobs->at(0)->num(), 1, 1,
				(float_t) (1), _phase);
		_bias_multiplier->set_variable(false);
	} else {
		o_blobs->at(0)->resize(s_blobs->at(0)->num(), _args->output_channel(),
				1, 1);
		_bias_multiplier->resize(1, s_blobs->at(0)->num(), 1, 1);
		_bias_multiplier->set_data(1.0);
	}
}

void equalized_inner_product_op::init_weights() {

	_w = create_param("w", _args->output_channel(), _args->channel(), 1, 1,
			_phase);

	_bias = create_param("bias", _args->output_channel(), 1, 1, 1, _phase);
}

void equalized_inner_product_op::check() {
	if (_args == NULL)
		LOG_FATAL("innerproduct data args cannot equal to NULL!");
	//output_channel > 0
	CHECK_GT_OP(_args->output_channel(), 0, "output_channel must > 0 vs %d",
			_args->output_channel());
}

void equalized_inner_product_op::op(blobs *&s_blobs_, blobs *&o_blobs_) {

	//blob *bias_multiplier = (blob*) _bias_multiplier;
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

	cacu_scalex(_w->s_data(), _w->count(), _w_mul);
	if(_is_use_bias)
		cacu_scalex(_bias->s_data(), _bias->count(), _b_mul);

	cacu_sgemm(TRANS, NOTRANS, _w->s_data(), _w->num(), _w->length(),
			s_blob_->s_data(), s_blob_->num(), (float_t) 1., o_blob_->s_data(),
			(float_t) 0.);
	//bias added
	if (_is_use_bias)
		for (int i = 0; i < s_blob_->num(); ++i) {
			cacu_saxpby(_bias->s_data(), (float_t) (1), o_blob_->p_data(i),
					(float_t) (1), _bias->count());
//			for(int c = 0; c < _w->num(); ++c){
//				if(FIND_FROM_VEC(*_w->upgrade_index(), c))
//					cacu_set_value(o_blob_->p_data(i) + c * o_blob_->channel_length(), 0, o_blob_->channel_length());
//			}
		}
	cacu_scalex(_w->s_data(), _w->count(), 1.0 / _w_mul);
	if(_is_use_bias)
		cacu_scalex(_bias->s_data(), _bias->count(), 1.0 / _b_mul);
#endif
}

void equalized_inner_product_op::grad(blobs *&s_blobs_, blobs *&o_blobs_) {

	blob *bias_multiplier = _bias_multiplier;

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

	cacu_scalex(_w->s_data(), _w->count(), _w_mul);
	if(_is_use_bias)
		cacu_scalex(_bias->s_data(), _bias->count(), _b_mul);

	//gradient propagation
	cacu_sgemm(NOTRANS, NOTRANS, _w->s_data(), _w->length(), _w->num(),
			o_blob_->s_diff(), o_blob_->num(), (float_t) 1., s_blob_->s_diff(),
			(float_t) 0.);
	//weights gradient
	cacu_sgemm(NOTRANS, TRANS, s_blob_->s_data(), s_blob_->length(),
			o_blob_->num(), o_blob_->s_diff(), _args->output_channel(),
			(float_t) 1., _w->s_diff(), (float_t) 1.);
	if (_is_use_bias)
		//bias gradient
		cacu_sgemv(NOTRANS, o_blob_->s_diff(), _args->output_channel(),
				bias_multiplier->s_data(), bias_multiplier->count(),
				(float_t) (1), _bias->s_diff(), (float_t) (1));


	cacu_scalex(_w->s_data(), _w->count(), 1.0 / _w_mul);
	cacu_scalex(_w->s_diff(), _w->count(), _w_mul);
	if(_is_use_bias){
		cacu_scalex(_bias->s_data(), _bias->count(), 1.0 / _b_mul);
		cacu_scalex(_bias->s_diff(), _bias->count(), _b_mul);
	}

#endif
}

void equalized_inner_product_op::load(std::ifstream& is) {
	_w->load(is);
	if (_is_use_bias)
		_bias->load(is);
}

void equalized_inner_product_op::save(std::ostream& os) {
	_w->serializa(os);
	if (_is_use_bias)
		_bias->serializa(os);
}

void equalized_inner_product_op::echo() {
	LOG_INFO("equalized inner product op: in(%d, %d, %d), out(%d, %d, %d)",
			s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
			s_blobs->at(0)->height(), o_blobs->at(0)->channel(),
			o_blobs->at(0)->width(), o_blobs->at(0)->height());
}

REGISTER_OP_CLASS_D_DP(equalized_inner_product_op);
REGISTER_OP_MACRO(equalized_inner_product_op, CACU_E_INNERPRODUCT);

}
