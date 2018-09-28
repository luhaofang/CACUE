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

#ifndef BATCH_NORMALIZE_OP_HPP_
#define BATCH_NORMALIZE_OP_HPP_

namespace cacu {

class batch_normalize_op: public operator_base {

public:

	batch_normalize_op(blobs *&data) :
			operator_base(data, CACU_BATCH_NORMALIZE) {
		_INIT_OP();
	}

	~batch_normalize_op() {

	}

	void initial() override {
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_oblobs();
			o_blobs->push_back(create_em_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height(), _phase));
			//save for train
			if (train == _phase)
				_x = create_em_opblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
						s_blobs->at(0)->width(), s_blobs->at(0)->height(), test);
			else
				_x = NULL;

			_dim_sum = create_em_opblob(1, s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
					s_blobs->at(0)->height(), test);
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(), _phase));
			//save for train
			_x = create_opblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(), test);

			_dim_sum = create_opblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), 1, 1, test);
#endif
			_moving_scalar = create_opblob(1, 1, 1, 1, test);
			_one = create_opblob(1, 1, 1, 1, 1, test);
			_mutipler = create_opblob(1, s_blobs->at(0)->channel_length(), 1, 1, 1.0,
					test);
			_mutipler->set_variable(false);
			_num_mutipler = create_opblob(1, s_blobs->at(0)->num(), 1, 1, 1.0, test);
			_num_mutipler->set_variable(false);
		} else {

			o_blobs->at(0)->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
				s_blobs->at(0)->height());
			//save for train

			_x->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
						s_blobs->at(0)->height());

			_dim_sum->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), 1, 1);

			_mutipler->resize(1, s_blobs->at(0)->channel_length(), 1, 1);
			_mutipler->set_data(1.0);
			_num_mutipler->resize(1, s_blobs->at(0)->num(), 1, 1);
			_num_mutipler->set_data(1.0);
		}
	}

	void init_weights() override {
		_scale = create_param("scale", s_blobs->at(0)->channel(), 1, 1, 1, _phase);
		_scale->set_init_type(constant, 1);
		_shift = create_param("shift", s_blobs->at(0)->channel(), 1, 1, 1, _phase);

		_mean = create_opblob(s_blobs->at(0)->channel(), 1, 1, 1, _phase);
		_var = create_opblob(s_blobs->at(0)->channel(), 1, 1, 1, _phase);

		//DO NOT reset in loops
		_history_mean = create_opblob(s_blobs->at(0)->channel(), 1, 1, 1, test);
		_history_mean->set_variable(false);
		_history_var = create_opblob(s_blobs->at(0)->channel(), 1, 1, 1, test);
		_history_var->set_variable(false);
		_std = create_opblob(s_blobs->at(0)->channel(), 1, 1, 1, _phase);
	}

	void check() override {
		//if phase is test, use global stat!
		if(_phase == test && use_global_stats == false)
			LOG_WARNING("BN use_global_stats set to \'false\' at test phase!");
	}

	void op(blobs *s_blobs_,blobs *o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;
		blob *dim_sum_ = (blob*) _dim_sum;
		em_blob *x_ = (em_blob*) _x;

		float_t m = (float_t) s_blob_->num() * s_blob_->width()
				* s_blob_->height();

		if (!use_global_stats) {
			float_t bias_correction_factor =
					m > (float_t) 1.0 ?
							(m) / (m - (float_t) 1.0) : (float_t) 1.0;

			dim_sum_->_RESET_DATA();
			for (int i = 0; i < s_blob_->num(); ++i)
				cacu_saxpy(s_blob_->p_data_d(i), 1, dim_sum_->s_data(),
						dim_sum_->count());
			cacu_sumbysize(BYWIDTH, dim_sum_->s_data(), dim_sum_->count(), 1,
					_mean->s_data(), 0, s_blob_->height() * s_blob_->width());
			cacu_scalex(_mean->s_data(), _mean->count(), ((float_t) 1.0 / m));

			dim_sum_->_RESET_DATA();
			for (int i = 0; i < s_blob_->num(); ++i) {
				cacu_ssxpy(_mean->s_data(), (float_t) (-1), _mean->count(),
						s_blob_->p_data_d(i), (float_t) (1), s_blob_->length(),
						o_blob_->p_data_d(i));
				//for saving space here we use o_data for container calculate x^2
				cacu_sqr(o_blob_->p_data_d(i), o_blob_->length(),
						o_blob_->p_data_d(i));
				cacu_saxpy(o_blob_->p_data_d(i), 1, dim_sum_->s_data(),
						dim_sum_->count());
			}
			cacu_sumbysize(BYWIDTH, dim_sum_->s_data(), dim_sum_->count(), 1,
					_var->s_data(), 0, s_blob_->height() * s_blob_->width());
			cacu_scalex(_var->s_data(), _var->count(), ((float_t) 1.0 / m));

			//cacu_scalex(_moving_scalar->s_data(), 1, moving_average_fraction);
			//cacu_saxpy(_one->s_data(), 1, _moving_scalar->s_data(), 1);
			//cacu_saxpby(_one->s_data(), (float_t)(1), _moving_scalar->s_data(), moving_average_fraction, _moving_scalar->count());

			//update history
			cacu_saxpby(_mean->s_data(), moving_average_fraction,
					_history_mean->s_data(), 1.0 - moving_average_fraction,
					_mean->count());
			cacu_saxpby(_var->s_data(), moving_average_fraction,
					_history_var->s_data(), 1.0 - moving_average_fraction,
					_var->count());

			cacu_stdbychannel(_var->s_data(), _std->count(), _std->s_data(),
					epsilon);
			for (int i = 0; i < s_blob_->num(); ++i) {
				cacu_ssxpy(_mean->s_data(), (float_t) (-1), _mean->count(),
						s_blob_->p_data_d(i), (float_t) (1), s_blob_->length(),
						o_blob_->p_data_d(i));
				cacu_cdxsize(o_blob_->p_data_d(i), o_blob_->length(),
						_std->s_data(), _std->count(), o_blob_->p_data_d(i));
				//save for train
				cacu_copy(o_blob_->p_data_d(i), o_blob_->length(),
						x_->p_data_d(i));
				cacu_cxsize(o_blob_->p_data_d(i), o_blob_->length(),
						_scale->s_data(), _scale->count(),
						o_blob_->p_data_d(i));
				cacu_ssxpy(_shift->s_data(), (float_t) (1), _shift->count(),
						o_blob_->p_data_d(i), (float_t) (1), o_blob_->length(),
						o_blob_->p_data_d(i));
				o_blob_->_sync(i);
				x_->_sync(i);
			}
		} else {
			//calculate unbiased estimate
			//cacu_cdxsize(_history_var->s_data(),_history_var->count(),_moving_scalar->s_data(),1,_var->s_data());
			//cacu_cdxsize(_history_mean->s_data(),_history_mean->count(),_moving_scalar->s_data(),1,_mean->s_data());

			cacu_copy(_history_var->s_data(), _history_var->count(),
					_var->s_data());
			cacu_copy(_history_mean->s_data(), _history_var->count(),
					_mean->s_data());

			cacu_stdbychannel(_var->s_data(), _std->count(), _std->s_data(),
					epsilon);
			for (int i = 0; i < s_blob_->num(); ++i) {
				cacu_ssxpy(_mean->s_data(), (float_t) (-1), _mean->count(),
						s_blob_->p_data_d(i), (float_t) (1), s_blob_->length(),
						o_blob_->p_data_d(i));
				cacu_cdxsize(o_blob_->p_data_d(i), o_blob_->length(),
						_std->s_data(), _std->count(), o_blob_->p_data_d(i));
				cacu_cxsize(o_blob_->p_data_d(i), o_blob_->length(),
						_scale->s_data(), _scale->count(),
						o_blob_->p_data_d(i));
				cacu_ssxpy(_shift->s_data(), (float_t) (1), _shift->count(),
						o_blob_->p_data_d(i), (float_t) (1), o_blob_->length(),
						o_blob_->p_data_d(i));
				o_blob_->_sync(i);
			}
		}
#else
		blob *o_blob_ = (blob*)o_blobs_->at(0);
		blob *s_blob_ = (blob*)s_blobs_->at(0);
		blob *dim_sum_ = (blob*)_dim_sum;
		blob *x_ = (blob*)_x;
		//cout << "bn:" << use_global_stats << endl;
		//cacu_print(s_blob_->s_data(), 10);
		float_t m = (float_t)s_blob_->num()*s_blob_->width()*s_blob_->height();
		if(!use_global_stats){

			//float_t bias_correction_factor = m > 1.0 ? (m) / (m - 1.0) : 1.0;

			//cacu_print(_mean->s_data(), _mean->count());
			//cacu_sumbysize(BYWIDTH, s_blob_->s_data(), s_blob_->count(),1, dim_sum_->s_data(),0, s_blob_->length()/s_blob_->channel());
			cacu_sgemv(TRANS, s_blob_->s_data(), _mutipler->count(), _mutipler->s_data(), dim_sum_->count(), (float_t)(1), dim_sum_->s_data(),0);
			//cacu_print(_mutipler->s_data(), _mean->count());
			//cacu_sumbysize(BYHEIGHT, dim_sum_->s_data(), s_blob_->channel() * s_blob_->num(), 1, _mean->s_data(), 0, s_blob_->channel());
			cacu_sgemv(NOTRANS, dim_sum_->s_data(), _mean->count(), _num_mutipler->s_data(), _num_mutipler->count(), (float_t)(1), _mean->s_data(), 0);
			//cacu_print(_mean->s_data(), _mean->count());
			cacu_scalex(_mean->s_data(), _mean->count(), (1.0 / m));

			for (int i = 0; i < s_blob_->num(); ++i)
			cacu_ssxpy(_mean->s_data(),(float_t)(-1),_mean->count(),s_blob_->p_data(i),(float_t)(1),s_blob_->length(),o_blob_->p_data(i));

			//for saving space here we use x_ for container calculate x^2
			cacu_sqr(o_blob_->s_data(), o_blob_->count(), x_->s_data());

			//cacu_sumbysize(BYWIDTH, x_->s_data(), o_blob_->count(), 1,dim_sum_->s_data(), 0, o_blob_->length()/o_blob_->channel());
			cacu_sgemv(TRANS, x_->s_data(), _mutipler->count(), _mutipler->s_data(), dim_sum_->count(), (float_t)(1), dim_sum_->s_data(), (float_t)(0));
			//cacu_sumbysize(BYHEIGHT, dim_sum_->s_data(), o_blob_->channel() * o_blob_->num(), 1, _var->s_data(), 0, o_blob_->channel());
			cacu_sgemv(NOTRANS, dim_sum_->s_data(), _var->count(), _num_mutipler->s_data(), _num_mutipler->count(), (float_t)(1), _var->s_data(), (float_t)(0));
			cacu_scalex(_var->s_data(), _var->count(), (1.0 / m));

			cacu_stdbychannel(_var->s_data(), _std->count(), _std->s_data(), epsilon);

			for (int i = 0; i < s_blob_->num(); ++i) {
				cacu_ssxpy(_mean->s_data(), (float_t)(-1), _mean->count(), s_blob_->p_data(i), (float_t)(1), s_blob_->length(), o_blob_->p_data(i));
				cacu_cdxsize(o_blob_->p_data(i), s_blob_->length(), _std->s_data(), _std->count(), o_blob_->p_data(i));
				//save for train
				cacu_copy(o_blob_->p_data(i),s_blob_->length(), x_->p_data(i));
				cacu_cxsize(o_blob_->p_data(i), s_blob_->length(), _scale->s_data(), _scale->count(), o_blob_->p_data(i));
				cacu_ssxpy(_shift->s_data(), (float_t)(1), _shift->count(), o_blob_->p_data(i), (float_t)(1), s_blob_->length(), o_blob_->p_data(i));
			}

			cacu_saxpby(_one->s_data(), (float_t)(1), _moving_scalar->s_data(), moving_average_fraction, _moving_scalar->count());

			//update history
			//cacu_saxpby(_mean->s_data(), (float_t)(1), _history_mean->s_data(), moving_average_fraction, _mean->count());
			//cacu_saxpby(_var->s_data(), bias_correction_factor, _history_var->s_data(), moving_average_fraction, _var->count());

			cacu_saxpby(_mean->s_data(), moving_average_fraction, _history_mean->s_data(), 1.0 - moving_average_fraction, _mean->count());
			cacu_saxpby(_var->s_data(), moving_average_fraction, _history_var->s_data(), 1.0 - moving_average_fraction, _var->count());
		}
		else{
			//calculate unbiased estimate
			//cacu_cdxsize(_history_var->s_data(),_history_var->count(),_moving_scalar->s_data(),1,_var->s_data());
			//cacu_cdxsize(_history_mean->s_data(),_history_mean->count(),_moving_scalar->s_data(),1,_mean->s_data());

			cacu_copy(_history_var->s_data(),_history_var->count(),_var->s_data());
			cacu_copy(_history_mean->s_data(),_history_var->count(),_mean->s_data());

			cacu_stdbychannel(_var->s_data(), _std->count(), _std->s_data(), epsilon);

			for (int i = 0; i < s_blob_->num(); ++i) {
				cacu_ssxpy(_mean->s_data(), (float_t)(-1), _mean->count(), s_blob_->p_data(i), (float_t)(1), s_blob_->length(), o_blob_->p_data(i));
				cacu_cdxsize(o_blob_->p_data(i), s_blob_->length(), _std->s_data(), _std->count(), o_blob_->p_data(i));
				cacu_cxsize(o_blob_->p_data(i), s_blob_->length(), _scale->s_data(), _scale->count(), o_blob_->p_data(i));
				cacu_ssxpy(_shift->s_data(), (float_t)(1), _shift->count(), o_blob_->p_data(i), (float_t)(1), s_blob_->length(), o_blob_->p_data(i));
			}
		}
		//cacu_print(o_blob_->s_data(), 10);
#endif
	}

	void grad(blobs *s_blobs_,blobs *o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;
		blob *dim_sum_ = (blob*) _dim_sum;
		em_blob *x_ = (em_blob*) _x;

		float_t *mean_data_, *mean_diff_;

		dim_sum_->_RESET_DATA();
		for (int i = 0; i < s_blob_->num(); ++i) {
			//calculate dl/x_
			cacu_cxsize(o_blob_->p_diff_d(i), o_blob_->length(),
					_scale->s_data(), _scale->count(), s_blob_->p_diff_d(i));
			cacu_saxpy(o_blob_->p_diff_d(i), 1, dim_sum_->s_data(),
					dim_sum_->count());
			//gradient of scale
			cacu_bn_gamma_grad(x_->p_data_d(i), o_blob_->p_diff_d(i), 1,
					o_blob_->length(), o_blob_->channel(), _scale->s_diff());
			s_blob_->_sync(i);
		}
		//gradient of shift
		cacu_sumbysize(BYWIDTH, dim_sum_->s_data(), dim_sum_->count(), 1,
				_shift->s_diff(), 0, s_blob_->height() * s_blob_->width());

		mean_data_ = _mean->s_data();
		mean_diff_ = _mean->s_diff();

		//calculate dl/std^2
		for (int i = 0; i < s_blob_->num(); ++i)
			cacu_bn_rou_grad(s_blob_->p_data_d(i), s_blob_->p_diff_d(i),
					mean_data_, _std->s_data(), 1, s_blob_->length(),
					s_blob_->channel(), _std->s_diff());

		//calculate dl/mu
		for (int i = 0; i < s_blob_->num(); ++i)
			cacu_bn_mu_grad(s_blob_->p_data_d(i), s_blob_->p_diff_d(i),
					mean_data_, _std->s_data(), _std->s_diff(), 1,
					s_blob_->length(), s_blob_->channel(), mean_diff_);

		//calculate dl/x
		for (int i = 0; i < s_blob_->num(); ++i) {
			cacu_bn_dx_grad(s_blob_->p_data_d(i), s_blob_->p_diff_d(i),
					mean_data_, _std->s_data(), _std->s_diff(), mean_diff_, 1,
					s_blob_->length(), s_blob_->channel(),
					s_blob_->p_diff_d(i));
			s_blob_->_sync(i);
		}

#else
		blob *o_blob_ = (blob*)o_blobs_->at(0);
		blob *s_blob_ = (blob*)s_blobs_->at(0);
		blob *dim_sum_ = (blob*)_dim_sum;
		blob *x_ = (blob*)_x;

		float_t *mean_data_,*mean_diff_;

		for(int i = 0; i < s_blob_->num(); ++i) {
			//calculate dl/x_
			cacu_cxsize(o_blob_->p_diff(i), s_blob_->length(), _scale->s_data(), _scale->count(), s_blob_->p_diff(i));
		}
		mean_data_ = _mean->s_data();
		mean_diff_ = _mean->s_diff();
		//calculate dl/std^2
		cacu_bn_rou_grad(s_blob_->s_data(), s_blob_->s_diff(), mean_data_, _std->s_data(), s_blob_->num(),s_blob_->length(),s_blob_->channel(), _std->s_diff());
		//calculate dl/mu
		cacu_bn_mu_grad(s_blob_->s_data(), s_blob_->s_diff(), mean_data_, _std->s_data(), _std->s_diff(), s_blob_->num(), s_blob_->length(), s_blob_->channel(), mean_diff_);
		//calculate dl/x
		cacu_bn_dx_grad(s_blob_->s_data(), s_blob_->s_diff(), mean_data_, _std->s_data(), _std->s_diff(), mean_diff_, s_blob_->num(), s_blob_->length(), s_blob_->channel(), s_blob_->s_diff());
		//gradient of scale
		cacu_bn_gamma_grad(x_->s_data(), o_blob_->s_diff(),s_blob_->num(), s_blob_->length(), s_blob_->channel(),_scale->s_diff());
		//gradient of shift
		//cacu_sumbysize(BYWIDTH, o_blob_->s_diff(), o_blob_->count(), 1, dim_sum_->s_data(), 0, o_blob_->length() / o_blob_->channel());
		//cacu_sumbysize(BYHEIGHT, dim_sum_->s_data(), s_blob_->channel() * s_blob_->num(), 1, _shift->s_diff(), 0, s_blob_->channel());
		cacu_sgemv(TRANS, o_blob_->s_diff(), _mutipler->count(), _mutipler->s_data(), dim_sum_->count(), (float_t)(1), dim_sum_->s_data(),0);
		cacu_sgemv(NOTRANS, dim_sum_->s_data(), _shift->count(), _num_mutipler->s_data(), _num_mutipler->count(), (float_t)(1), _shift->s_diff(), 1);
#endif
	}

	void load(std::ifstream& is) override {

		_history_mean->load(is);
		//_history_mean->set_data(0);
		_history_var->load(is);

		//_moving_scalar->load(is);
		//_history_var->set_data(0);
		//blob_base *blob = cacu_allocator::create_blob(1, 1, 1, 1, test);
		//blob->load(is);
		_scale->load(is);
		_shift->load(is);
	}

	void save(std::ostream& os) override {

		_history_mean->serializa(os);
		_history_var->serializa(os);

		_scale->serializa(os);
		_shift->serializa(os);
	}

	void echo() override
	{
		LOG_INFO("create batch_normalize op:");
		LOG_INFO(
				"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height());
	}

	inline weight* scale() {
		return _scale;
	}

	inline weight* shift() {
		return _shift;
	}

	inline void set_scale_init_type(param_init_type _type,
			float_t value = 0.0) {
		set_param_init_type(_type, _scale, value);
	}

	inline void set_scale_init_type(param_init_type _type, float_t mean,
				float_t std) {
		set_param_init_type(_type, _scale, std);
		cacu_sdxsize<float_t>(_scale->s_data(),_scale->count(),mean,1.0,_scale->s_data());
	}

	inline void set_shift_init_type(param_init_type _type,
			float_t value = 0.0) {
		set_param_init_type(_type, _shift, value);
	}

	float_t moving_average_fraction = 0.9;

	float_t epsilon = 0.00001;

	inline void set_phase(phase_type phase_) {
		_phase = phase_;
		if(_scale->update()){
			if (train == _phase)
				use_global_stats = false;
			else
				use_global_stats = true;
		}
	}

	inline void set_is_use_global_stats(bool use_global_stats_)
	{
		//LOG_INFO("Use global stats: %d",use_global_stats_);
		use_global_stats = use_global_stats_;
	}

private:

	weight *_scale = NULL;

	weight *_shift = NULL;

	blob *_mean = NULL;

	blob *_var = NULL;

	blob *_std = NULL;

	blob *_history_mean = NULL;

	blob *_history_var = NULL;

	blob_base *_dim_sum = NULL;

	blob_base *_x = NULL;

	blob *_moving_scalar = NULL;

	blob *_one = NULL;
	blob *_mutipler = NULL;
	blob *_num_mutipler = NULL;

	bool use_global_stats = false;


};
}


#endif
