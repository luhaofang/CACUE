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


namespace mycnn{

	class batch_normal_op : public operator_base
	{

	public:

		batch_normal_op(blob *&data, args *&args_) : operator_base((blob_base*&)data, args_){

			check();
			o_blob = cacu_allocator::create_blob(data->num(), data->channel(), data->width(), data->height());

			_scale = new weight("scale", data->channel(), 1, 1, 1, data->phrase());
			_shift = new weight("shift", data->channel(), 1, 1, 1, data->phrase());

			_mean = cacu_allocator::create_blob(data->channel(), 1, 1, 1);
			_var = cacu_allocator::create_blob(data->channel(), 1, 1, 1);		

			_history_mean = cacu_allocator::create_blob(data->channel(), 1, 1, 1);
			_history_var = cacu_allocator::create_blob(data->channel(), 1, 1, 1);

			_std = cacu_allocator::create_blob(data->channel(), 1, 1, 1);

			op();
		};

		~batch_normal_op(){

			delete (blob *)o_blob;

			delete (blob *)_mean;
			delete (blob *)_var;

			delete (blob *)_history_mean;
			delete (blob *)_history_var;

			delete (blob *)_std;

			delete _scale;
			delete _shift;
		};

		virtual const void check() override{
			return;
		}

		virtual const void op() override {
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;

			float_t m = (float_t)s_blob_->num()*s_blob_->width()*s_blob_->height();
			float_t bias_correction_factor = m > (float_t)1.0 ? (m) / (m - (float_t)1.0) : (float_t)1.0;

			if (!use_global_stats)
			{
				vec_t dim_sum(s_blob_->num() * s_blob_->channel());

				cacu_sumbysize(BYWIDTH, s_blob_->s_data(), s_blob_->count(), &dim_sum[0], s_blob_->width()*s_blob_->height());
				cacu_sumbysize(BYHEIGHT, &dim_sum[0], s_blob_->channel()*s_blob_->num(), _mean->s_data(), s_blob_->channel());
				cacu_sxsize(_mean->s_data(), _mean->count(), ((float_t)1.0 / m), _mean->s_data());

				//for saving space here we use o_data for container calculate x^2
				cacu_sqr(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
				cacu_sumbysize(BYWIDTH, o_blob_->s_data(), o_blob_->count(), &dim_sum[0], o_blob_->width()*o_blob_->height());
				cacu_sumbysize(BYHEIGHT, &dim_sum[0], o_blob_->channel()*o_blob_->num(), _var->s_data(), o_blob_->channel());
				cacu_sxsize(_var->s_data(), _var->count(), ((float_t)1.0 / m), _var->s_data());

				cacu_saxpby(_mean->s_data(), (float_t)-1.0, _var->s_data(), (float_t)1.0, _mean->count());

				//update history
				cacu_saxpby(_mean->s_data(), ((float_t)(1) - moving_average_fraction), _history_mean->s_data(), moving_average_fraction, _mean->count());
				cacu_saxpby(_var->s_data(), ((float_t)(1) - moving_average_fraction)*bias_correction_factor, _history_var->s_data(), moving_average_fraction, _var->count());

				cacu_stdbychannel(_var->s_data(), _std->count(), _std->s_data(), epsilon);

				for (int i = 0; i < s_blob_->num(); ++i){
					cacu_ssxpy(_mean->s_data(), (float_t)(-1), _mean->count(), s_blob_->p_data(i), (float_t)(1), s_blob_->length(), o_blob_->p_data(i));
					cacu_cdxsize(o_blob_->p_data(i), o_blob_->length(), _std->s_data(), _std->count(), o_blob_->p_data(i));

					cacu_cxsize(o_blob_->p_data(i), o_blob_->length(), _scale->s_data(), _scale->count(), o_blob_->p_data(i));
					cacu_ssxpy(_shift->s_data(), (float_t)(1), _shift->count(), o_blob_->p_data(i), (float_t)(1), o_blob_->length(), o_blob_->p_data(i));
				}
			}
			else
			{
				cacu_stdbychannel(_history_var->s_data(), _std->count(), _std->s_data(), epsilon);

				for (int i = 0; i < s_blob_->num(); ++i){
					cacu_ssxpy(_mean->s_data(), (float_t)(-1), _mean->count(), s_blob_->p_data(i), (float_t)(1), s_blob_->length(), o_blob_->p_data(i));
					cacu_cdxsize(o_blob_->p_data(i), o_blob_->length(), _std->s_data(), _std->count(), o_blob_->p_data(i));

					cacu_cxsize(o_blob_->p_data(i), o_blob_->length(), _scale->s_data(), _scale->count(), o_blob_->p_data(i));
					cacu_ssxpy(_shift->s_data(), (float_t)(1), _shift->count(), o_blob_->p_data(i), (float_t)(1), o_blob_->length(), o_blob_->p_data(i));
				}
			}

			echo();
			return;
		}

		virtual const void grad(const solver_base *&solver_base) override{

		}

		virtual const void load(std::ifstream& is){

		}

		virtual const void save(std::ostream& os){

		}

		virtual const void echo()
		{

			//LOG_INFO("%f", _mean->s_data()[0]);
		}

		inline weight* scale(){ return _scale; }

		inline weight* shift(){ return _shift; }

		bool use_global_stats = false;

		float_t moving_average_fraction = 0.9f;

		float_t epsilon = 0.00001f;

	private:

		weight *_scale;

		weight *_shift;

		blob *_mean;
		
		blob *_var;

		blob *_std;

		blob *_history_mean;

		blob *_history_var;

	};
};
