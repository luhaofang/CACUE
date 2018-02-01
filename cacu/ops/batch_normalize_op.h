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

#pragma once


namespace cacu{

	class batch_normalize_op : public operator_base
	{

	public:

		batch_normalize_op(blob_base *&data, data_args *&args_) : operator_base(data, args_, CACU_BATCH_NORMALIZE){

			check();
			initial(data,args_);
			init_weights(data,args_);
			echo();
		};

		~batch_normalize_op(){

			delete _mean;
			delete _var;

			delete _history_mean;
			delete _history_var;

			delete _std;

			delete _dim_sum;

			delete _moving_scalar;

			delete _mutipler;
			delete _num_mutipler;

			if(_x != NULL)
				delete _x;
		};

		virtual const void initial(blob_base *&data, data_args *&args_) override{
#if __USEMBEDDING__ == ON
			o_blob = create_em_oblob(data->num(), data->channel(), data->height(),data->width(), _phase);
			//save for train
			if(train == _phase)
				_x = cacu_allocator::create_em_blob(data->num(), data->channel(), data->height(), data->width(), test);
			else
				_x = NULL;

			_dim_sum = cacu_allocator::create_blob(1, data->channel(), data->height(), data->width(), test);
#else
			o_blob = create_oblob(data->num(), data->channel(), data->height(), data->width(), _phase);
			//save for train
			if(train == _phase)
				_x = cacu_allocator::create_blob(data->num(), data->channel(), data->height(),data->width(), test);
			else
				_x = NULL;

			_dim_sum = cacu_allocator::create_blob(data->num(), data->channel(), 1, 1, test);
#endif


			_moving_scalar = cacu_allocator::create_blob(1, 1, 1, 1, test);

			_one = cacu_allocator::create_blob(1, 1, 1, 1, 1,test);

			_mutipler = cacu_allocator::create_blob(1,data->channel_length(),1,1,1.0,test);
			_num_mutipler = cacu_allocator::create_blob(1,data->num(),1,1,1.0,test);
		}

		virtual const void init_weights(blob_base *&data, data_args *&args_) override{
			_scale = create_param("scale", data->channel(), 1, 1, 1, _phase);
			_scale->set_init_type(constant,1);
			_shift = create_param("shift", data->channel(), 1, 1, 1, _phase);
			_shift->set_lr(2);

			_mean = cacu_allocator::create_blob(data->channel(), 1, 1, 1, _phase);
			_var = cacu_allocator::create_blob(data->channel(), 1, 1, 1,_phase);

			_history_mean = cacu_allocator::create_blob(data->channel(), 1, 1, 1,test);
			_history_var = cacu_allocator::create_blob(data->channel(), 1, 1, 1,test);

			_std = cacu_allocator::create_blob(data->channel(), 1, 1, 1,_phase);
		}

		virtual const void check() override{
			//training for batch_size
			if(train == _phase){
				//CHECK_GT_OP(s_blob->num(), 1 ,"batch size for training must > 1 vs %d",s_blob->num());
				use_global_stats = false;
			}
			else
				use_global_stats = true;
		}

		virtual const void op() override {

#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blob;
			blob *dim_sum_ = (blob*)_dim_sum;
			em_blob *x_ = (em_blob*)_x;

			float_t m = (float_t)s_blob_->num()*s_blob_->width()*s_blob_->height();

			if (!use_global_stats)
			{
				float_t bias_correction_factor = m > (float_t)1.0 ? (m) / (m - (float_t)1.0) : (float_t)1.0;

				dim_sum_->_RESET_DATA();
				for(int i = 0 ; i < s_blob_->num(); ++i)
					cacu_saxpy(s_blob_->p_data_d(i), 1, dim_sum_->s_data(), dim_sum_->count());
				cacu_sumbysize(BYWIDTH, dim_sum_->s_data(), dim_sum_->count(), 1, _mean->s_data(), 0, s_blob_->height() * s_blob_->width());
				cacu_scalex(_mean->s_data(), _mean->count(), ((float_t)1.0 / m));

				dim_sum_->_RESET_DATA();
				for (int i = 0; i < s_blob_->num(); ++i){
					cacu_ssxpy(_mean->s_data(),(float_t)(-1),_mean->count(),s_blob_->p_data_d(i),(float_t)(1),s_blob_->length(),o_blob_->p_data_d(i));
					//for saving space here we use o_data for container calculate x^2
					cacu_sqr(o_blob_->p_data_d(i), o_blob_->length(), o_blob_->p_data_d(i));
					cacu_saxpy(o_blob_->p_data_d(i), 1, dim_sum_->s_data(), dim_sum_->count());
				}
				cacu_sumbysize(BYWIDTH, dim_sum_->s_data(), dim_sum_->count(), 1, _var->s_data(), 0, s_blob_->height() * s_blob_->width());
				cacu_scalex(_var->s_data(), _var->count(), ((float_t)1.0 / m));

				//cacu_scalex(_moving_scalar->s_data(), 1, moving_average_fraction);
				//cacu_saxpy(_one->s_data(), 1, _moving_scalar->s_data(), 1);
				//cacu_saxpby(_one->s_data(), (float_t)(1), _moving_scalar->s_data(), moving_average_fraction, _moving_scalar->count());

				//update history
				cacu_saxpby(_mean->s_data(), moving_average_fraction, _history_mean->s_data(), 1.0 - moving_average_fraction, _mean->count());
				cacu_saxpby(_var->s_data(), moving_average_fraction, _history_var->s_data(), 1.0 - moving_average_fraction, _var->count());

				cacu_stdbychannel(_var->s_data(), _std->count(), _std->s_data(), epsilon);
				for (int i = 0; i < s_blob_->num(); ++i){
					cacu_ssxpy(_mean->s_data(), (float_t)(-1), _mean->count(), s_blob_->p_data_d(i), (float_t)(1), s_blob_->length(), o_blob_->p_data_d(i));
					cacu_cdxsize(o_blob_->p_data_d(i), o_blob_->length(), _std->s_data(), _std->count(), o_blob_->p_data_d(i));
					//save for train
					cacu_copy(o_blob_->p_data_d(i),o_blob_->length(),x_->p_data_d(i));
					cacu_cxsize(o_blob_->p_data_d(i), o_blob_->length(), _scale->s_data(), _scale->count(), o_blob_->p_data_d(i));
					cacu_ssxpy(_shift->s_data(), (float_t)(1), _shift->count(), o_blob_->p_data_d(i), (float_t)(1), o_blob_->length(), o_blob_->p_data_d(i));
					o_blob_->_sync(i);
					x_->_sync(i);
				}
			}
			else{
				//calculate unbiased estimate
				//cacu_cdxsize(_history_var->s_data(),_history_var->count(),_moving_scalar->s_data(),1,_var->s_data());
				//cacu_cdxsize(_history_mean->s_data(),_history_mean->count(),_moving_scalar->s_data(),1,_mean->s_data());

				cacu_copy(_history_var->s_data(),_history_var->count(),_var->s_data());
				cacu_copy(_history_mean->s_data(),_history_var->count(),_mean->s_data());

				cacu_stdbychannel(_var->s_data(), _std->count(), _std->s_data(), epsilon);
				for (int i = 0; i < s_blob_->num(); ++i){
					cacu_ssxpy(_mean->s_data(), (float_t)(-1), _mean->count(), s_blob_->p_data_d(i), (float_t)(1), s_blob_->length(), o_blob_->p_data_d(i));
					cacu_cdxsize(o_blob_->p_data_d(i), o_blob_->length(), _std->s_data(), _std->count(), o_blob_->p_data_d(i));
					cacu_cxsize(o_blob_->p_data_d(i), o_blob_->length(), _scale->s_data(), _scale->count(), o_blob_->p_data_d(i));
					cacu_ssxpy(_shift->s_data(), (float_t)(1), _shift->count(), o_blob_->p_data_d(i), (float_t)(1), o_blob_->length(), o_blob_->p_data_d(i));
					o_blob_->_sync(i);
				}
			}
#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;
			blob *dim_sum_ = (blob*)_dim_sum;
			blob *x_ = (blob*)_x;

			float_t m = (float_t)s_blob_->num()*s_blob_->width()*s_blob_->height();
			if (!use_global_stats)
			{
				float_t bias_correction_factor = m > 1.0 ? (m) / (m - 1.0) : 1.0;

				//cacu_print(_mean->s_data(), _mean->count());
				//cacu_sumbysize(BYWIDTH, s_blob_->s_data(), s_blob_->count(),1, dim_sum_->s_data(),0, s_blob_->length()/s_blob_->channel());
				cacu_sgemv(TRANS, s_blob_->s_data(), _mutipler->count(), _mutipler->s_data(), dim_sum_->count(), (float_t)(1), dim_sum_->s_data(),0);
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

				LOG_DEBUG("");
				cuda_print(_var->s_data(),3);

				for (int i = 0; i < s_blob_->num(); ++i){
					cacu_ssxpy(_mean->s_data(), (float_t)(-1), _mean->count(), s_blob_->p_data(i), (float_t)(1), s_blob_->length(), o_blob_->p_data(i));
					cacu_cdxsize(o_blob_->p_data(i), o_blob_->length(), _std->s_data(), _std->count(), o_blob_->p_data(i));
					//save for train
					cacu_copy(o_blob_->p_data(i),o_blob_->length(), x_->p_data(i));
					cacu_cxsize(o_blob_->p_data(i), o_blob_->length(), _scale->s_data(), _scale->count(), o_blob_->p_data(i));
					cacu_ssxpy(_shift->s_data(), (float_t)(1), _shift->count(), o_blob_->p_data(i), (float_t)(1), o_blob_->length(), o_blob_->p_data(i));
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

				for (int i = 0; i < s_blob_->num(); ++i){
					cacu_ssxpy(_mean->s_data(), (float_t)(-1), _mean->count(), s_blob_->p_data(i), (float_t)(1), s_blob_->length(), o_blob_->p_data(i));
					cacu_cdxsize(o_blob_->p_data(i), o_blob_->length(), _std->s_data(), _std->count(), o_blob_->p_data(i));
					cacu_cxsize(o_blob_->p_data(i), o_blob_->length(), _scale->s_data(), _scale->count(), o_blob_->p_data(i));
					cacu_ssxpy(_shift->s_data(), (float_t)(1), _shift->count(), o_blob_->p_data(i), (float_t)(1), o_blob_->length(), o_blob_->p_data(i));
				}
			}
#endif
		}

		virtual const void grad() override{

#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blob;
			blob *dim_sum_ = (blob*)_dim_sum;
			em_blob *x_ = (em_blob*)_x;

			float_t *mean_data_,*mean_diff_;

			dim_sum_->_RESET_DATA();
			for(int i = 0 ; i < s_blob_->num(); ++i){
				//calculate dl/x_
				cacu_cxsize(o_blob_->p_diff_d(i), o_blob_->length(), _scale->s_data(), _scale->count(), s_blob_->p_diff_d(i));
				cacu_saxpy(o_blob_->p_diff_d(i), 1, dim_sum_->s_data(), dim_sum_->count());
				//gradient of scale
				cacu_bn_gamma_grad(x_->p_data_d(i), o_blob_->p_diff_d(i), 1, o_blob_->length(), o_blob_->channel(),_scale->s_diff());
				s_blob_->_sync(i);
			}
			//gradient of shift
			cacu_sumbysize(BYWIDTH, dim_sum_->s_data(), dim_sum_->count(), 1, _shift->s_diff(), 0, s_blob_->height() * s_blob_->width());

			mean_data_ = _mean->s_data();
			mean_diff_ = _mean->s_diff();

			//calculate dl/std^2
			for(int i = 0 ; i < s_blob_->num(); ++i)
				cacu_bn_rou_grad(s_blob_->p_data_d(i), s_blob_->p_diff_d(i), mean_data_, _std->s_data(), 1,s_blob_->length(),s_blob_->channel(), _std->s_diff());

			//calculate dl/mu
			for(int i = 0 ; i < s_blob_->num(); ++i)
				cacu_bn_mu_grad(s_blob_->p_data_d(i), s_blob_->p_diff_d(i), mean_data_, _std->s_data(), _std->s_diff(), 1, s_blob_->length(), s_blob_->channel(), mean_diff_);

			//calculate dl/x
			for(int i = 0 ; i < s_blob_->num(); ++i){
				cacu_bn_dx_grad(s_blob_->p_data_d(i), s_blob_->p_diff_d(i), mean_data_, _std->s_data(), _std->s_diff(), mean_diff_, 1, s_blob_->length(), s_blob_->channel(), s_blob_->p_diff_d(i));
				s_blob_->_sync(i);
			}

#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;
			blob *dim_sum_ = (blob*)_dim_sum;
			blob *x_ = (blob*)_x;

			float_t *mean_data_,*mean_diff_;

			for(int i = 0 ; i < s_blob_->num(); ++i){
				//calculate dl/x_
				cacu_cxsize(o_blob_->p_diff(i), o_blob_->length(), _scale->s_data(), _scale->count(), s_blob_->p_diff(i));
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
			cacu_bn_gamma_grad(x_->s_data(), o_blob_->s_diff(),o_blob_->num(), o_blob_->length(), o_blob_->channel(),_scale->s_diff());
			//gradient of shift
			//cacu_sumbysize(BYWIDTH, o_blob_->s_diff(), o_blob_->count(), 1, dim_sum_->s_data(), 0, o_blob_->length() / o_blob_->channel());
			//cacu_sumbysize(BYHEIGHT, dim_sum_->s_data(), s_blob_->channel() * s_blob_->num(), 1, _shift->s_diff(), 0, s_blob_->channel());
			cacu_sgemv(TRANS, o_blob_->s_diff(), _mutipler->count(), _mutipler->s_data(), dim_sum_->count(), (float_t)(1), dim_sum_->s_data(),0);
			cacu_sgemv(NOTRANS, dim_sum_->s_data(), _shift->count(), _num_mutipler->s_data(), _num_mutipler->count(), (float_t)(1), _shift->s_diff(), 0);
#endif
		}

		virtual const void load(std::ifstream& is) override {

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

		virtual const void save(std::ostream& os) override {

			_history_mean->serializa(os);
			_history_var->serializa(os);


			_scale->serializa(os);
			_shift->serializa(os);
		}

		virtual const void echo() override
		{
			LOG_INFO("create batch_normalize op:");
			LOG_INFO("channel: %d, input_dim: %d, output_channel: %d, output_dim: %d",s_blob->channel(),s_blob->height(),o_blob->channel(),o_blob->height());
		}

		inline virtual const void LOOP_INIT_DATA_() override
		{
			o_blob->_RESET_DATA();

			_scale->_RESET_DIFF();
			_shift->_RESET_DIFF();

			_mean->_RESET_DATA();
			_var->_RESET_DATA();

			_std->_RESET_DATA();

			if(_x != NULL)
				_x->_RESET_DATA();

			_dim_sum->_RESET_DATA();

		}

		inline weight* scale(){ return _scale; }

		inline weight* shift(){ return _shift; }

		inline void set_scale_init_type(param_init_type _type,float_t value = 0.0){set_param_init_type(_type, _scale, value);}

		inline void set_shift_init_type(param_init_type _type,float_t value = 0.0){set_param_init_type(_type, _shift, value);}

		bool use_global_stats = true;

		float_t moving_average_fraction = 0.9;

		float_t epsilon = 0.00001;

		inline virtual const void set_phase(phase_type phase_) override {
			_phase = phase_;
			if(train == _phase)
				use_global_stats = false;
			else
				use_global_stats = true;
		}

	private:

		weight *_scale;

		weight *_shift;

		blob *_mean;

		blob *_var;

		blob *_std;

		blob *_history_mean;

		blob *_history_var;

		blob_base *_dim_sum;

		blob_base *_x;

		blob *_moving_scalar;

		blob *_one;
		blob *_mutipler;
		blob *_num_mutipler;

	};
};
