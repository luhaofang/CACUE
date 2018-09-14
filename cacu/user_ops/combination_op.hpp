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

#ifndef COMBINATION_OP_HPP_
#define COMBINATION_OP_HPP_


namespace cacu {

class combination_op: public operator_base {

public:

	combination_op(blobs *&data) :
			operator_base(data, CACU_COMBINATION) {
		_INIT_OP();
	}

	~combination_op() {

	}

	void initial() override {

		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_oblobs();
			o_blobs->push_back(create_em_oblob(s_blobs->at(0)->num(),
					s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
					s_blobs->at(0)->height(), _phase));
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num()*s_blobs->size(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height(), _phase));
#endif
		} else {
			o_blobs->at(0)->resize(s_blobs->at(0)->num()*s_blobs->size(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height());
		}
	}

	void init_weights() override {
		return;
	}

	void check() override {

		CHECK_GE_OP(s_blobs->size(), 2, "blobs size must >= 2 vs %d",
				s_blobs->size());
		for (size_t i = 1; i < s_blobs->size(); ++i) {
			CHECK_EQ_OP(s_blobs->at(i)->count(), s_blobs->at(0)->count(),
					"blob count must be equal %d vs %d",
					s_blobs->at(i)->count(), s_blobs->at(0)->count());
		}

	}

	void op(blobs *s_blobs_,blobs *o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);

		for (int i = 0; i < o_blob_->num(); ++i) {
			for (unsigned int j = 0; j < (s_blobs)->size(); ++j) {
				em_blob *s_blob_ = (em_blob *) s_blobs->at(j);
				cacu_saxpy(s_blob_->p_data_d(i), (float_t) 1,
						o_blob_->p_data_d(i), o_blob_->length());
			}
			o_blob_->_sync(i);
		}
#else
		blob *o_blob_ = (blob*)o_blobs_->at(0);
		for (unsigned int j = 0; j < s_blobs_->size(); ++j) {
			blob *s_blob_ = (blob *)s_blobs_->at(j);
			cacu_copy(s_blob_->s_data(),s_blob_->count(), o_blob_->p_data(j*s_blobs_->at(0)->num()));
		}
#endif
	}

	void grad(blobs *s_blobs_,blobs *o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);

		for (unsigned int j = 0; j < (s_blobs)->size(); ++j) {
			em_blob *s_blob_ = (em_blob *) s_blobs->at(j);
			cacu_copy_cpu(o_blob_->s_diff(), o_blob_->count(),
					s_blob_->s_diff());
		}
#else
		blob *o_blob_ = (blob*)o_blobs_->at(0);

		for (unsigned int j = 0; j < (s_blobs_)->size(); ++j) {
			blob *s_blob_ = (blob *)s_blobs_->at(j);
			cacu_copy(o_blob_->p_diff(j*s_blobs_->at(0)->num()),s_blob_->count(),s_blob_->s_diff());
		}
#endif
	}

	void load(std::ifstream& is) override {
		return;
	}

	void save(std::ostream& os) override {
		return;
	}

	void echo() override {
		LOG_INFO("create combination op:");
		LOG_INFO(
				"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
				s_blobs->at(0)->height(), o_blobs->at(0)->channel(), o_blobs->at(0)->width(),
				o_blobs->at(0)->height());
	}


private:

};
}


#endif
