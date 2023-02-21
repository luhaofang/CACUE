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

#ifndef SIGN_ACTIV_OP_HPP_
#define SIGN_ACTIV_OP_HPP_

namespace cacu {

class sign_activ_op: public activate_base_op {

public:

	sign_activ_op(blobs *&data) :
			activate_base_op(data, CACU_SIGMOID_SIGN) {
		_INIT_OP();
	}

	~sign_activ_op() {

	}

	void initial() override {
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_oblobs();
			o_blobs->push_back(create_em_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->height(), s_blobs->at(0)->width(), _phase));
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height(), _phase));
			_temp = create_opblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height(), test);
#endif
		} else {
			o_blobs->at(0)->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
					s_blobs->at(0)->height());
			_temp->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
								s_blobs->at(0)->width(), s_blobs->at(0)->height());
		}
	}

	void op(blobs *&s_blobs_,blobs *&o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_sigmoid_cpu(s_blob_->s_data(), o_blob_->count(),
				o_blob_->s_data());

#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);

		cacu_htanh(s_blob_->s_data(), _temp->count(), _temp->s_data());
		cacu_ssx(_temp->s_data(), _temp->count(),o_blob_->s_data());

#endif
		//echo();
	}

	void grad(blobs *&s_blobs_,blobs *&o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_sigmoid_grad_cpu(o_blob_->s_data(), o_blob_->s_diff(),
				s_blob_->count(), s_blob_->s_diff());

#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);

		cacu_htanh_grad(_temp->s_data(), s_blob_->s_data(), s_blob_->count(), s_blob_->s_diff());
		cacu_saxpy(_temp->s_data(),1,s_blob_->s_diff(),_temp->count());
		cacu_ssx(o_blob_->s_diff(), o_blob_->count(), s_blob_->s_diff());
#endif
		//echo();

	}

	void echo() override {
		LOG_INFO("create sign_sigmoid op:");
		LOG_INFO(
				"channel: %d, input_dim: %d, output_channel: %d, output_dim: %d",
				s_blobs->at(0)->channel(), s_blobs->at(0)->height(), o_blobs->at(0)->channel(),
				o_blobs->at(0)->height());
	}

private:

	blob *_temp = NULL;

};
}


#endif
