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

#include "ce_loss_op.h"
#include "../math/math.h"


namespace cacu {

	ce_loss_op::ce_loss_op(blobs *&data) :
			loss_base_op(data, CACU_CE_LOSS) {
		_INIT_OP();
	}

	ce_loss_op::~ce_loss_op() {

	}

	void ce_loss_op::initial() {
		_loss = 0.0;
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_blobs();
			o_blobs = create_em_oblob(s_blobs->at(0)->num(),
					s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
					s_blobs->at(0)->height(), train);
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num(),s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height(),train));
#endif
		} else {
			o_blobs->at(0)->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height());
		}
	}

	void ce_loss_op::check() {
		//check blob size
		CHECK_EQ_OP((int)s_blobs->size(), 2, "source blob size = 2 vs %d !",
				(int)s_blobs->size());
	}

	void ce_loss_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {

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
//		blob *s_blob_ = s_blobs_->asblob(0);
		bin_blob *labels_ = s_blobs_->asbinblob(1);

		cacu_cross_entropy_multi(o_blob_->s_data(),o_blob_->num(),o_blob_->channel(), o_blob_->width(), o_blob_->height(),labels_->s_data(),o_blob_->s_diff());
#endif
#if __USEMBEDDING__ == ON
		cacu_copy_cpu(o_blob_->s_diff(), 1, &_loss);
#else
#if __USE_DEVICE__ == ON
		cuda_copy2host(&_loss, o_blob_->s_diff(), 1);
#else
		cacu_copy(o_blob_->s_diff(), 1 ,&_loss);
#endif
#endif
		_loss *= normalizer();
		_loss /= o_blob_->channel_length();
	}

	void ce_loss_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {

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

		cacu_cross_entropy_multi_grad(o_blob_->s_data(),o_blob_->num(),o_blob_->channel(), o_blob_->width(), o_blob_->height(),labels_->s_data(),s_blob_->s_diff());
#endif
	}

REGISTER_OP_CLASS_D(ce_loss_op);
REGISTER_OP_MACRO(ce_loss_op, CACU_CE_LOSS);

}

