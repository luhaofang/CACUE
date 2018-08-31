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

#ifndef SIGMOID_WITH_LOSS_OP_HPP_
#define SIGMOID_WITH_LOSS_OP_HPP_

namespace cacu {

class sigmoid_with_loss_op: public operator_base {

public:

	sigmoid_with_loss_op(blobs *&data) :
			operator_base(data, CACU_SIGMOID_LOSS) {
		_INIT_OP();
	}

	~sigmoid_with_loss_op() {

	}

	void initial()  {
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

	void init_weights()  {
		return;
	}

	void check()  {
		//check blob size
		CHECK_GT_OP(s_blobs->size(), 1, "source blob size > 1 vs %d !",
				s_blobs->size());
		//check blob size
		CHECK_EQ_OP(s_blobs->at(0)->channel(), 1, "source data must = 1 vs %d !",
				s_blobs->at(0)->channel());
		CHECK_EQ_OP(s_blobs->at(0)->count(), s_blobs->at(1)->count(), "source data must equal %d vs %d !",
					s_blobs->at(0)->count(),s_blobs->at(1)->count());
	}

	void op(blobs *s_blobs_,blobs *o_blobs_)  {

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
		blob *o_blob_ = (blob*)o_blobs_->at(0);
		blob *s_blob_ = (blob*)s_blobs_->at(0);
		bin_blob *labels_ = (bin_blob*)s_blobs_->at(1);

		cacu_sigmoid(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());

		vec_t _temp(o_blobs_->at(0)->count());
		vec_i _target(o_blobs_->at(0)->count());

#if __USE_DEVICE__ == ON
		cuda_copy2host(&_temp[0], s_blob_->s_data(), s_blob_->count());
		cuda_copy2host(&_target[0], labels_->s_data(), s_blob_->count());
#else
		cacu_copy(s_blob_->s_data(), s_blob_->count(), &_temp[0]);
		cacu_sdxsize(labels_->s_data(), s_blob_->count(), 0, 1, &_target[0]);
#endif

		for(int i = 0 ; i< s_blob_->count(); ++i)
		{
			_loss -= _temp[i] * (_target[i] - (_temp[i] >= 0.0)) - log(1.0 + exp(_temp[i] - 2 * _temp[i] * (_temp[i] >= 0.0)));
 		}
		_loss = max(_loss, float_t(_MIN_FLT_));
		_loss *= normalizer();
		_loss /= o_blob_->channel_length();
#endif
	}

	void grad(blobs *s_blobs_,blobs *o_blobs_)  {

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
		blob *o_blob_ = (blob*)o_blobs_->at(0);
		blob *s_blob_ = (blob*)s_blobs_->at(0);
		bin_blob *labels_ = (bin_blob*)s_blobs_->at(1);

		vec_i _target(o_blobs_->at(0)->count());
#if __USE_DEVICE__ == ON
		cuda_copy2host(&_target[0], labels_->s_data(), s_blob_->count());
#else
		cacu_sdxsize(labels_->s_data(), s_blob_->count(), 0, 1, &_target[0]);
#endif
		//CE LOSS BACK PROPGATION
		cacu_copy(o_blob_->s_data(), s_blob_->count(), s_blob_->s_diff());
		for(int i = 0; i < s_blob_->count(); ++i)
		{
			if(_target[i] == 1)
				cacu_sdxsize(s_blob_->s_diff()+i, 1, (float_t)-1.0, (float_t)1.0, s_blob_->s_diff()+i);
		}
		cacu_scalex(s_blob_->s_diff(), s_blob_->count(), normalizer() * _loss_weight);

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
		LOG_INFO("loss : %f", _loss);
		if(_loss_weight != 1.0)
			LOG_INFO("weighted loss : %f", _loss * _loss_weight);
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

	inline void set_loss_weight(float_t weight_)
	{
		_loss_weight = weight_;
	}

private:

	float_t _loss = 0.0;

	float_t _loss_weight = 1.0;
};
}


#endif /* SIGMOID_WITH_LOSS_OP_HPP_ */
