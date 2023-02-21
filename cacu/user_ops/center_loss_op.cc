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

#include "center_loss_op.h"
#include "../math/math.h"

namespace cacu {

	center_loss_op::center_loss_op(blobs *&data, op_args *&o_args_) :
			loss_base_op(data, o_args_, CACU_CENTER_LOSS) {
		_INIT_OP();
	}

	center_loss_op::~center_loss_op() {

	}

	void center_loss_op::initial() {
		_loss = 0.0;
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_blobs();
			o_blobs = create_em_oblob(s_blobs->at(0)->num(),
					s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
					s_blobs->at(0)->height(), train);
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num(), (int)_o_args->at(0), 1, 1,train));
			_x_norm = create_opblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(), test);
			_class_center = create_opblob((int)_o_args->at(0), s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(), 0, test);
			_class_center->set_variable(false);
			_temp = create_opblob(1, s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(), train);
#endif
		} else {
			o_blobs->at(0)->resize(s_blobs->at(0)->num(), (int)_o_args->at(0), 1, 1);
			_class_center->resize((int)_o_args->at(0), s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height());
			_x_norm->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height());
			_temp->resize(1, s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height());
		}
	}

	void center_loss_op::check() {
		if(_o_args == NULL)
			LOG_FATAL("center loss op args cannot equal to NULL!");
		//check blob size
		CHECK_GT_OP((int)s_blobs->size(), 1, "source blob size > 1 vs %d !",
				(int)s_blobs->size());
	}

	void center_loss_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {

		_loss = 0.0;

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);
		em_bin_blob *labels_ = (em_bin_blob*) s_blobs->at(1);

		cacu_softmax_cpu(s_blob_->s_data(), s_blob_->num(), s_blob_->channel(),
				o_blob_->width(), s_blob_->height(), o_blob_->s_data());
		cacu_cross_entropy_cpu(o_blob_->s_data(), o_blob_->num(),
				o_blob_->length(), labels_->s_data(), o_blob_->s_diff());

#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);
		bin_blob *labels_ = s_blobs_->asbinblob(1);

		// _temp(o_blobs_->at(0)->count());
		vec_i _target(labels_->count());
		vec_t temp(o_blob_->length());

#if __USE_DEVICE__ == ON
		//cuda_copy2host(&_temp[0], s_blob_->s_data(), s_blob_->count());
		cuda_copy2host(&_target[0], labels_->s_data(), labels_->count());
#else
		for(int i = 0 ; i < labels_->count(); ++i)
			_target[i] = labels_->s_data()[i];
#endif
		for(int i = 0 ; i < s_blob_->num(); ++i){
			cacu_copy(s_blob_->p_data(i), s_blob_->length(), _x_norm->p_data(i));
			cacu_l2_normalization(_x_norm->p_data(i), s_blob_->length(), 1e-8);
			cacu_saxpby(_x_norm->p_data(i), _moving_average_fraction, _class_center->p_data(_target[i]), 1.0 - _moving_average_fraction, _class_center->length());
		}

		for(int i = 0 ; i < s_blob_->num(); ++i){
			for(int c = 0; c < _class_center-> num(); ++c){
				cacu_copy(_x_norm->p_data(i), _temp->count(), _temp->s_data());
				cacu_saxpby(_class_center->p_data(c), -1.0, _temp->s_data(), 1.0, _class_center->length());
				cacu_sqr(_temp->s_data(), _temp->count(), _temp->s_data());
//				if(c == _target[i]){
//					cacu_sumbysize(BYWIDTH, _temp->s_data(), _temp->count(), _balance_margin, o_blob_->p_data(i) + c, (float_t)0, _temp->count());
//				}
//				else{
//					cacu_sumbysize(BYWIDTH, _temp->s_data(), _temp->count(), -(1.0 - _balance_margin) * (1.0 / (_class_center-> num() - 1)), o_blob_->p_data(i) + c, (float_t)0, _temp->count());
//				}
			}
		}
#if __USE_DEVICE__ == ON
		cacu_sumbysize_cpu(BYWIDTH, o_blob_->s_data_cpu(), o_blob_->count(),
				(float_t)(1.0 / o_blob_-> count()), &_loss, (float_t)0, o_blob_->count());
#else
		cacu_sumbysize(BYWIDTH, o_blob_->s_data(), o_blob_->count(),
				(float_t)(1.0 / o_blob_-> count()), &_loss, (float_t)0, o_blob_->count());
#endif
//		LOG_DEBUG("%f", _loss);
//		_loss = expf(_loss);
		_loss = max(_loss, float_t(_MIN_FLT_));
		_loss *= normalizer();
#endif
	}

	void center_loss_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);
		em_bin_blob *labels_ = (em_bin_blob*) s_blobs->at(1);

		//CE LOSS BACK PROPGATION
		for (int i = 0; i < s_blob_->count(); ++i) {
			cacu_isaxb(o_blob_->p_data_d(i), s_blob_->length(), (float_t) 1,
					labels_->p_data_d(i), (float_t) -1, s_blob_->p_diff_d(i));
			cacu_scalex(s_blob_->p_diff_d(i), s_blob_->length(), normalizer());
			s_blob_->_sync(i);
		}
#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);
		bin_blob *labels_ = s_blobs_->asbinblob(1);

		vec_i _target(labels_->count());

#if __USE_DEVICE__ == ON
		//cuda_copy2host(&_temp[0], s_blob_->s_data(), s_blob_->count());
		cuda_copy2host(&_target[0], labels_->s_data(), labels_->count());
#else
		for(int i = 0 ; i < labels_->count(); ++i)
			_target[i] = labels_->s_data()[i];
#endif
		for(int i = 0 ; i< s_blob_->num(); ++i){
			for(int c = 0; c < _class_center->num(); ++c){
				cacu_copy(_x_norm->p_data(i), _temp->count(), _temp->s_data());
				cacu_saxpby(_class_center->p_data(c), -1.0, _temp->s_data(), 1.0, _class_center->length());
				//cacu_copy(o_blob_->p_data(i) + c, 1, o_blob_->p_diff(i) + c);
				o_blob_->set_pdiff(1, i);
//				if(c == _target[i])
//					cacu_scalex(o_blob_->p_diff(i) + c, 1, _balance_margin);
//				else
//					cacu_scalex(o_blob_->p_diff(i) + c, 1, -(1.0 - _balance_margin) * (1.0 / (_class_center-> num() - 1)));
				cacu_cxsize(_temp->s_data(), _temp->count(), o_blob_->p_diff(i) + c, 1, _temp->s_diff());
				cacu_saxpby(_temp->s_diff(), (float_t)2, s_blob_->p_diff(i), (float_t)1, _temp->count());
			}
			cacu_l2_normalization_grad(s_blob_->p_data(i), s_blob_->length(), 1e-8, s_blob_->p_diff(i), s_blob_->p_diff(i));
		}
		cacu_scalex(s_blob_->s_diff(), s_blob_->count(), normalizer());
		//cacu_print(s_blob_->s_diff(), s_blob_->count());
#endif
	}

REGISTER_OP_CLASS_D_OP(center_loss_op);
REGISTER_OP_MACRO(center_loss_op, CACU_CENTER_LOSS);

}

 /* CENTER_LOSS_OP_HPP_ */
