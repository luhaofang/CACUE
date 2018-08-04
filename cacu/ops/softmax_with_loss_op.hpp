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

#ifndef SOFTMAX_WITH_LOSS_OP_HPP_
#define SOFTMAX_WITH_LOSS_OP_HPP_

namespace cacu {

class softmax_with_loss_op: public operator_base {

public:

	softmax_with_loss_op(blobs *&data, data_args *&args_) :
			operator_base(data, args_, CACU_SOFTMAX_LOSS) {
		check();

		initial();
		init_weights();

		_loss = 0.0;

		//echo();
	}

	~softmax_with_loss_op() {
	}

	void initial()  {
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

	void init_weights()  {
		return;
	}

	void check()  {
		//check blob size
		CHECK_GT_OP(s_blobs->size(), 1, "source blob size > 1 vs %d !",
				s_blobs->size());
		//check blob size
		CHECK_GT_OP(s_blobs->at(0)->channel(), 1, "source data must > 1 vs %d !",
				s_blobs->at(0)->channel());
	}

	void op()  {

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
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blobs->at(0);
		bin_blob *labels_ = (bin_blob*)s_blobs->at(1);

		cacu_softmax(s_blob_->s_data(), s_blob_->num(), s_blob_->channel(), s_blob_->width(), s_blob_->height(), o_blob_->s_data());
		//cacu_cross_entropy(o_blob_->s_data(),o_blob_->num(),o_blob_->length(),labels_->s_data(),o_blob_->s_diff());
		//LOG_DEBUG("%d,%d,%d",o_blob_->width(), o_blob_->height(),labels_->count());
		//CHECK_EQ_OP(o_blob_->channel_length(),labels_->count(), "%d,%d", o_blob_->count(),labels_->count());
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

	void grad()  {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);
		em_bin_blob *labels_ = (em_bin_blob*) s_blobs->at(1);

		//CE LOSS BACK PROPGATION
		for (int i = 0; i < s_blob_->num(); ++i) {
			cacu_isaxb(o_blob_->p_data_d(i), s_blob_->length(), (float_t) 1,
					labels_->p_data_d(i), (float_t) -1, s_blob_->p_diff_d(i));
			cacu_scalex(s_blob_->p_diff_d(i), s_blob_->length(), normalizer());
			s_blob_->_sync(i);
		}
#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blobs->at(0);
		bin_blob *labels_ = (bin_blob*)s_blobs->at(1);

		//CE LOSS BACK PROPGATION
		for (int i = 0; i < s_blob_->num(); ++i)
		{
			cacu_isaxb(o_blob_->p_data(i),s_blob_->channel(),s_blob_->width(),s_blob_->height(),(float_t)1,labels_->p_data(i),(float_t)-1, s_blob_->p_diff(i));
			cacu_scalex(s_blob_->p_diff(i), s_blob_->length(), normalizer() * _loss_weight / o_blob_->channel_length());
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

	inline void LOOP_INIT_DATA_() 
	{
		o_blob->_RESET_DATA();
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


#endif
