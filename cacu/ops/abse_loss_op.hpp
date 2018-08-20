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

#ifndef ABSE_LOSS_OP_HPP_
#define ABSE_LOSS_OP_HPP_

namespace cacu {

class abse_loss_op: public operator_base {

public:

	abse_loss_op(blobs *&data) :
			operator_base(data, CACU_ABSE_LOSS) {

		_INIT_OP();
	}

	~abse_loss_op() {

	}

	void initial()  {
		_loss = 0.0;
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_oblobs();
			o_blobs->push_back(create_em_oblob(s_blobs->at(0)->num(),
					s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
					s_blobs->at(0)->height(), train));
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num(),s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height(),train));
			_direction = create_opblob(s_blobs->at(0)->num(),s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height(),test);
#endif
		} else {

			o_blobs->at(0)->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
				s_blobs->at(0)->width(), s_blobs->at(0)->height());
			_direction->resize(s_blobs->at(0)->num(),s_blobs->at(0)->channel(),
				s_blobs->at(0)->width(), s_blobs->at(0)->height());
		}
	}

	void init_weights()  {
		return;
	}

	void check()  {
		//check blob size
		CHECK_EQ_OP(s_blobs->at(0)->count(), s_blobs->at(1)->count(),
				"source data must equal %d vs %d !", s_blobs->at(0)->count(),
				s_blobs->at(1)->count());
		s_blobs->at(0)->_CHECK_SIZE_EQ(s_blobs->at(1));
	}

	void op()  {

		_loss = 0.0;

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob->at(0);
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
		blob *o_blob_ = (blob*)o_blobs->at(0);
		blob *s_blob1_ = (blob*)s_blobs->at(0);
		blob *s_blob2_ = (blob*)s_blobs->at(1);

		//cacu_print(s_blob_->s_data(), 100);
		cacu_copy(s_blob2_->s_data(), s_blob2_->count(), o_blob_->s_data());
#if __USE_DEVICE__ == ON
		float_t *pdata_cpu = o_blob_->s_data_cpu();
#else
		float_t *pdata_cpu = o_blob_->s_data();
#endif
		for(int i = 0; i < s_blob1_->num(); ++i) {
			if(pdata_cpu[i * o_blob_->length()] != -1.0)
			cacu_saxpby(s_blob1_->p_data(i), 1, o_blob_->p_data(i), -1,
					s_blob1_->length());
			else
			cacu_saxpby(s_blob1_->p_data(i), 0, o_blob_->p_data(i), 0,
					s_blob1_->length());
		}
		cacu_abs(o_blob_->s_data(), o_blob_->count(), o_blob_->s_diff());
		cacu_sumbysize(BYWIDTH, o_blob_->s_diff(), o_blob_->count(), normalizer(),
				o_blob_->s_diff(), 0, o_blob_->count());
#endif

#if __USE_DEVICE__ == ON
		cuda_copy2host(&_loss, o_blob_->s_diff(), 1);
#else
		cacu_copy(o_blob_->s_diff(), 1 ,&_loss);
#endif

	}

	void grad()  {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_copy_cpu(o_blob_->s_data(), o_blob_->count(), s_blob_->s_diff());
		cacu_scalex_cpu(s_blob_->s_diff(), s_blob_->count(), normalizer());

#else
		blob *o_blob_ = (blob*)o_blobs->at(0);
		blob *s_blob1_ = (blob*)s_blobs->at(0);
		blob *s_blob2_ = (blob*)s_blobs->at(1);

		_direction->set_data(1);
		cacu_abs_grad(o_blob_->s_data(), s_blob1_->s_diff(), s_blob1_->count(),_direction->s_data());
		cacu_scalex(s_blob1_->s_diff(), s_blob1_->count(), normalizer() * _loss_weight);
		_direction->set_data(-1);
		cacu_abs_grad(o_blob_->s_data(), s_blob2_->s_diff(), s_blob2_->count(),_direction->s_data());
		cacu_scalex(s_blob2_->s_diff(), s_blob2_->count(), normalizer() * _loss_weight);
		//cacu_print(s_blob_->s_diff(),s_blob_->count());
#endif
	}

	void load(std::ifstream& is)  {
		return;
	}

	void save(std::ostream& os)  {
		return;
	}

	void echo()
	{
		LOG_INFO("abse loss : %f", _loss);
		if (_loss_weight != 1.0)
			LOG_INFO("weighted abse loss : %f", _loss * _loss_weight);
	}

	inline void LOOP_INIT_DATA_()
	{
		o_blobs->_RESET_DATA();
	}

	inline void set_phase(phase_type phase_)  {
		_phase = phase_;
	}

	float_t normalizer() {
		blob_base* blob_ = s_blobs->at(0);
		return ((float_t) (1) / blob_->num());
	}

	inline float_t loss() {
		return _loss;
	}

	inline void set_loss_weight(float_t weight_) {
		_loss_weight = weight_;
	}

private:

	float_t _loss = 0.0;

	blob* _direction = NULL;

	float_t _loss_weight = 1.0;
};
}


#endif
