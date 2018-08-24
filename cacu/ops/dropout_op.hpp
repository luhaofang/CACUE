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

#ifndef DROPOUT_OP_HPP_
#define DROPOUT_OP_HPP_

namespace cacu {

class dropout_op: public operator_base {

public:

	dropout_op(blobs *&data, op_args *&args_) :
			operator_base(data, args_, CACU_DROPOUT) {
		_INIT_OP();
	}

	~dropout_op() {

	}

	void initial()  {

		_ratio = _o_args->at(0);
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blob = s_blob;
			_rand_vect = create_em_opblob(s_blob->num(), s_blob->channel(),
					s_blob->width(), s_blob->height(), test);
#else
			o_blobs = s_blobs;
			_rand_vect = create_opblob(s_blobs->at(0)->num(),s_blobs->at(0)->channel(),s_blobs->at(0)->width(),s_blobs->at(0)->height(), test);
#endif
		} else {
			o_blobs->_NEED_MOTIFY();
			_rand_vect->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height());
		}

	}

	void init_weights()  {
		return;
	}

	void check()  {
		if(_o_args == NULL)
			LOG_FATAL("dropout op args cannot equal to NULL!");
		CHECK_GE_OP(_o_args->at(0), 0.0,
			"ratio must be a positive decimal larger than 0 vs %f!",
			_o_args->at(0));
		CHECK_LE_OP(_o_args->at(0), 1.0,
			"ratio must be a positive decimal smaller than 1 vs %f!",
			_o_args->at(0));
	}

	void op()  {

		float_t scale_ = 1.0 / (1 - _ratio);

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;
		em_blob *rand_vect_ = (em_blob*) _rand_vect;

		if (train == _phase) {
			for (int i = 0; i < s_blob_->num(); ++i) {
				rand_vector(rand_vect_->p_data_d(i), rand_vect_->length(),
						_ratio);
				cacu_ssx(rand_vect_->p_data_d(i), o_blob_->length(),
						o_blob_->p_data_d(i));
				cacu_scalex(o_blob_->p_data_d(i), o_blob_->length(), scale_);
				o_blob_->_sync(i);
				rand_vect_->_sync(i);
			}
		}
#else
		blob *o_blob_ = (blob*)o_blobs->at(0);
		blob *rand_vect_ = (blob*)_rand_vect;

		if(train == _phase)
		{
			rand_vector(rand_vect_->s_data(),rand_vect_->count(), _ratio);
			//cacu_output(o_blob_->s_data(),o_blob_->count(),"/home/seal/1.txt");
			cacu_ssx(rand_vect_->s_data(), o_blob_->count(), o_blob_->s_data());
			//cacu_output(o_blob_->s_data(),o_blob_->count(),"/home/seal/2.txt");
			cacu_scalex(o_blob_->s_data(), o_blob_->count(), scale_);
		}
#endif
	}

	void grad()  {

		float_t scale_ = 1.0 / (1 - _ratio);

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;
		em_blob *rand_vect_ = (em_blob*) _rand_vect;

		if (train == _phase) {
			for (int i = 0; i < s_blob_->num(); ++i) {
				//ratio's scale implementation
				cacu_ssx(rand_vect_->p_data_d(i), s_blob_->length(),
						s_blob_->p_diff_d(i));
				cacu_scalex(s_blob_->p_diff_d(i), o_blob_->length(), scale_);
				s_blob_->_sync(i);
			}
		}
#else
		blob *o_blob_ = (blob*)o_blobs->at(0);
		blob *s_blob_ = (blob*)s_blobs->at(0);
		blob *rand_vect_ = (blob*)_rand_vect;

		if(train == _phase)
		{
			//ratio's scale implementation
			cacu_ssx(rand_vect_->s_data(), s_blob_->count(), s_blob_->s_diff());
			cacu_scalex(s_blob_->s_diff(), o_blob_->count(), scale_);
		}
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
		LOG_INFO("create dropout op:");
		LOG_INFO(
				"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height());
	}


	inline void set_phase(phase_type phase_)  {
		_phase = phase_;
	}

private:

	blob *_rand_vect = NULL;

	float_t _ratio = 0.5;
};
}


#endif
