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

#ifndef MSE_LOSS_OP_HPP_
#define MSE_LOSS_OP_HPP_

namespace cacu {

class mse_loss_op: public operator_base {

public:

	mse_loss_op(blobs *&data) :
			operator_base(data, CACU_MSE_LOSS) {
		check();

		initial();
		init_weights();

		_loss = (float_t*) malloc(sizeof(float_t));
		_loss[0] = 0;
		echo();
	}

	~mse_loss_op() {
		free(_loss);
	}

	virtual const void initial() override {
		if (o_blob == NULL) {
#if __USEMBEDDING__ == ON
			o_blob = create_em_oblob(s_blobs->at(0)->num(),
					s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
					s_blobs->at(0)->height(), train);
#else
			o_blob = create_oblob(s_blobs->at(0)->num(),s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),train);
#endif
		} else {
			o_blob->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height());
		}
	}

	virtual const void init_weights() override {
		return;
	}

	virtual const void check() override {
		//check blob size
		CHECK_EQ_OP(s_blobs->at(0)->count(), s_blobs->at(1)->count(),
				"source data must equal %d vs %d !", s_blobs->at(0)->count(),
				s_blobs->at(1)->count());
	}

	virtual const void op() override {

		_loss[0] = 0.0;

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);
		em_blob *labels_ = (em_blob*) s_blobs->at(1);

		cacu_copy_cpu(labels_->s_data(), labels_->count(), o_blob_->s_data());

		for (int i = 0; i < s_blob_->num(); ++i) {
			if (o_blob_->p_data(i) != -1.0)
				cacu_saxpby_cpu(s_blob_->p_data(i), 1, o_blob_->p_data(i), -1,
						s_blob_->length());
			else
				cacu_saxpby_cpu(s_blob_->p_data(i), 0, o_blob_->p_data(i), 0,
						s_blob_->length());
		}
		cacu_sqr_cpu(o_blob_->s_data(), o_blob_->count(), o_blob_->s_diff());
		cacu_scalex_cpu(o_blob_->s_diff(), o_blob_->count(), 0.5);
		cacu_sumbysize_cpu(BYWIDTH, o_blob_->s_diff(), o_blob_->count(), 1,
				_loss, 0, o_blob_->count());
#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blobs->at(0);
		blob *labels_ = (blob*)s_blobs->at(1);

		//cacu_print(s_blob_->s_data(), 100);
		cacu_copy(labels_->s_data(), labels_->count(), o_blob_->s_data());
#if __USE_DEVICE__ == ON
		float_t *pdata_cpu = o_blob_->s_data_cpu();
#else
		float_t *pdata_cpu = o_blob_->s_data();
#endif
		for(int i = 0; i < s_blob_->num(); ++i) {
			if(pdata_cpu[i * o_blob_->length()] != -1.0)
			cacu_saxpby(s_blob_->p_data(i), 1, o_blob_->p_data(i), -1,
					s_blob_->length());
			else
			cacu_saxpby(s_blob_->p_data(i), 0, o_blob_->p_data(i), 0,
					s_blob_->length());
		}
		cacu_sqr(o_blob_->s_data(), o_blob_->count(), o_blob_->s_diff());
		cacu_scalex(o_blob_->s_diff(), o_blob_->count(), 0.5);
		cacu_sumbysize(BYWIDTH, o_blob_->s_diff(), o_blob_->count(), normalizer(),
				o_blob_->s_diff(), 0, o_blob_->count());
#endif

#if __USE_DEVICE__ == ON
		cuda_copy2host(_loss, o_blob_->s_diff(), 1);
#else
		cacu_copy(o_blob_->s_diff(), 1 ,_loss);
#endif

	}

	virtual const void grad() override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_copy_cpu(o_blob_->s_data(), o_blob_->count(), s_blob_->s_diff());
		cacu_scalex_cpu(s_blob_->s_diff(), s_blob_->count(), normalizer());

#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob1_ = (blob*)s_blobs->at(0);
		blob *s_blob2_ = (blob*)s_blobs->at(1);

		cacu_copy(o_blob_->s_data(),o_blob_->count(),s_blob1_->s_diff());
		cacu_scalex(s_blob1_->s_diff(), s_blob1_->count(), normalizer());
		cacu_copy(o_blob_->s_data(), o_blob_->count(), s_blob2_->s_diff());
		cacu_scalex(s_blob2_->s_diff(), s_blob2_->count(), -normalizer());
		//cacu_print(s_blob_->s_diff(),s_blob_->count());
#endif
	}

	virtual const void load(std::ifstream& is) override {
		return;
	}

	virtual const void save(std::ostream& os) override {
		return;
	}

	virtual const void echo() override
	{
		LOG_INFO("mse loss : %f", _loss[0]);
		if (_loss_weight != 1.0)
			LOG_INFO("weighted mse loss : %f", _loss[0] * _loss_weight);
	}

	inline virtual const void LOOP_INIT_DATA_() override
	{
		o_blob->_RESET_DATA();
	}

	inline virtual const void set_phase(phase_type phase_) override {
		_phase = phase_;
	}

	float_t normalizer() {
		blob_base* blob_ = s_blobs->at(0);
		return ((float_t) (1) / blob_->num());
	}

	inline float_t loss() {
		return _loss[0];
	}

	inline void set_loss_weight(float_t weight_) {
		_loss_weight = weight_;
	}

private:

	float_t *_loss;

	float_t _loss_weight = 1.0;
};
}


#endif