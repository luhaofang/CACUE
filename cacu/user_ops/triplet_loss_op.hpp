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

#ifndef TRIPLET_LOSS_OP_HPP_
#define TRIPLET_LOSS_OP_HPP_

namespace cacu {

class triplet_loss_op: public loss_base_op {

public:

	triplet_loss_op(blobs *&data, op_args *&args_) :
			loss_base_op(data, args_, CACU_TRIPLET_LOSS) {
		_INIT_OP();
	}

	~triplet_loss_op() {

	}

	void initial() override {

		_loss = 0.0;
		_margin = _o_args->at(0);

		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			_sets = create_em_opblob(s_blobs->at(0)->num(),
					1, 1, 1, test);
#else
			_sets = create_opblob(s_blobs->at(0)->num(),1, 1, 1,test);

#endif
		} else {
			_sets->resize(s_blobs->at(0)->num(),1, 1, 1);
		}

	}

	void check() override {
		if(_o_args == NULL)
			LOG_FATAL("tripletloss op args cannot equal to NULL!");
		CHECK_GT_OP(_o_args->at(0), 0, "margin must > 0 vs %d", _o_args->at(0));
		CHECK_EQ_OP(s_blobs->size(), 3, "number of blobs must equal to 3 vs %d", s_blobs->size());
		CHECK_EQ_OP(s_blobs->at(0)->count(), s_blobs->at(1)->count(), "input blob size must equal to %d vs %d", s_blobs->at(0)->count(), s_blobs->at(1)->count());
		CHECK_EQ_OP(s_blobs->at(1)->count(), s_blobs->at(2)->count(), "input blob size must equal to %d vs %d", s_blobs->at(1)->count(), s_blobs->at(2)->count());
	}

	void op(blobs *s_blobs_,blobs *o_blobs_) override {
#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;
		em_bin_blob *index_ = (em_bin_blob*) _index;
		em_blob *x_ = (em_blob*) _x;
		if (_phase == train) {
			for (int i = 0; i < s_blob_->num(); ++i) {
				cacu_copy(s_blob_->p_data_d(i), s_blob_->length(),
						x_->s_data());
				cacu_row_max_pooling(x_->s_data(), x_->count(),
						o_blob_->length(), o_blob_->p_data_d(i));
				cacu_row_max_pooling_index(s_blob_->p_data_d(i),
						s_blob_->length(), o_blob_->length(),
						o_blob_->p_data_d(i), index_->p_data_d(i));
				index_->_sync(i);
			}
		} else
			for (int i = 0; i < s_blob_->num(); ++i) {
				cacu_row_max_pooling(s_blob_->p_data_d(i), s_blob_->length(),
						o_blob_->length(), o_blob_->p_data_d(i));
				o_blob_->_sync(i);
			}
#else

		blob *a_blob_ = (blob*)s_blobs_->at(0);
		blob *p_blob_ = (blob*)s_blobs_->at(1);
		blob *n_blob_ = (blob*)s_blobs_->at(2);

		float_t a_p_dist_ = 0;
		float_t a_n_dist_ = 0;
		float_t dist_ = 0;

		_loss = 0.0;

		cacu_copy(p_blob_->s_data(),p_blob_->count(),p_blob_->s_diff());
		cacu_saxpby(a_blob_->s_data(),(float_t)(1.0), p_blob_->s_diff(), (float_t)(-1.0), a_blob_->count());
		cacu_copy(n_blob_->s_data(),n_blob_->count(),n_blob_->s_diff());
		cacu_saxpby(a_blob_->s_data(),(float_t)(1.0), n_blob_->s_diff(), (float_t)(-1.0), a_blob_->count());

		for(int i = 0; i< a_blob_->num(); ++i){
			cacu_sgemv(NOTRANS, p_blob_->p_diff(i), 1, p_blob_->s_diff(), p_blob_->length(), (float_t)(1.0), &a_p_dist_, (float_t)(1.0));
			cacu_sgemv(NOTRANS, n_blob_->p_diff(i), 1, n_blob_->s_diff(), n_blob_->length(), (float_t)(1.0), &a_n_dist_, (float_t)(1.0));

			dist_ = _margin + a_p_dist_ - a_n_dist_;
			if(dist_ > 0.0)
				_sets->set_pdata(1,i);
			else
				_sets->set_pdata(0,i);
			_loss += (dist_ > 0.0) ? dist_ : 0.0;
		}
#endif
	}

	void grad(blobs *s_blobs_,blobs *o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;
		em_bin_blob *index_ = (em_bin_blob*) _index;
		for (int i = 0; i < s_blob_->num(); ++i) {
			cacu_row_max_pooling_grad(o_blob_->p_diff_d(i), o_blob_->length(),
					s_blob_->p_diff_d(i), index_->p_data_d(i));
			s_blob_->_sync(i);
		}
#else
		blob *a_blob_ = (blob*)s_blobs_->at(0);
		blob *p_blob_ = (blob*)s_blobs_->at(1);
		blob *n_blob_ = (blob*)s_blobs_->at(2);

		cacu_copy(n_blob_->s_data(), a_blob_->count(), a_blob_->s_diff());
		cacu_saxpby(p_blob_->s_data(),(float_t)(-1.0), a_blob_->s_diff(), (float_t)(1.0), a_blob_->count());

		cacu_copy(p_blob_->s_data(), a_blob_->count(), p_blob_->s_diff());
		cacu_saxpby(a_blob_->s_data(),(float_t)(-1.0), p_blob_->s_diff(), (float_t)(1.0), a_blob_->count());

		cacu_copy(a_blob_->s_data(), a_blob_->count(), n_blob_->s_diff());
		cacu_saxpby(n_blob_->s_data(),(float_t)(-1.0), n_blob_->s_diff(), (float_t)(1.0), a_blob_->count());

		for(int i = 0; i < a_blob_->num(); ++i)
		{
			cacu_cxsize(a_blob_->p_diff(i),a_blob_->length(),_sets->p_data(i), 1, a_blob_->p_diff(i));
			cacu_cxsize(p_blob_->p_diff(i),p_blob_->length(),_sets->p_data(i), 1, p_blob_->p_diff(i));
			cacu_cxsize(n_blob_->p_diff(i),n_blob_->length(),_sets->p_data(i), 1, n_blob_->p_diff(i));
		}

		cacu_scalex(a_blob_->s_diff(), a_blob_->count(), normalizer() * _loss_weight );
		cacu_scalex(n_blob_->s_diff(), a_blob_->count(), normalizer() * _loss_weight );
		cacu_scalex(p_blob_->s_diff(), a_blob_->count(), normalizer() * _loss_weight );
#endif
	}

private:

	blob* _sets = NULL;

	float_t _margin = 0.0;

};
}


#endif
