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

	class solver_base
	{

	public:

		solver_base(network *&net_)
		{
			_net = net_;

			_temp = cacu_allocator::create_blobs();
			for(int i = 0; i < _net->op_count(); ++i)
			{
				operator_base* op_ = _net->get_op(i);
				for(int j = 0; j < op_->weights_size(); ++j)
				{
					blob *temp_w = op_->get_weight(j)->copy_create(test);
					_temp->push_back(temp_w);
				}
			}
		};

		virtual ~solver_base(){


		};

		inline void set_weight_decay(float_t weight_decay_){ _global_weight_decay = weight_decay_;}

		inline void set_lr(float_t lr_){ _global_lr = lr_ ;}

		inline void set_regularize(regularize_type type_){_regularize = type_;}

		inline float_t weight_decay(){ return _global_weight_decay;}

		inline float_t lr(){ return _global_lr;}

		inline regularize_type regularize(){ return _regularize;}

		/*
		 * where weight_index denote the weight's id in sovler's vector
		 */
		virtual const void update_weight(weight* w_, int weight_index_) = 0;

		void crop_grad(blob* g_){};

		inline void train_iter()
		{

			_net->predict();
			for(int i = _net->op_count() - 1 ; i >= 0; --i)
			{
				_net->get_op(i)->grad();
			}

			int weight_index_ = 0;
			//update weights
			for(int i = 0 ; i < _net->op_count();++i)
			{
				operator_base* op_ = _net->get_op(i);
				for(int j = 0; j < op_->weights_size(); ++j)
				{
					update_weight(op_->get_weight(j),weight_index_);
					weight_index_++;
				}
			}
		}

		/**
		 * change global_lr by rate after several training iterations
		 *
		 */
		void set_lr_iter(float_t lr_rate_)
		{
			this->_global_lr *= lr_rate_;
		}

	protected:

		float_t _global_lr = 1.0;

		float_t _global_weight_decay = 0.004f;

		regularize_type _regularize = L2;

		network *_net;

		/**
		 * add regular to gradient
		 * where i is the index of _w
		 */
		void __REGULARIZE__(weight *w_ , int weight_index_)
		{
			float_t weight_decay_ = w_->decay() * _global_weight_decay;
			blob* temp = (blob*)_temp->at(weight_index_);
			switch(_regularize)
			{
			case L1 :
				rand_vector(temp->s_data(),temp->count(),1);
				cacu_saxpy(temp->s_data(), weight_decay_, w_->s_diff(), w_->count());
				break;
			case L2 :
				cacu_saxpy(w_->s_data(), weight_decay_, w_->s_diff(), w_->count());
				break;
			default:
				break;
			}
		}

		/**
		 * normalize gradient
		 * where i is the index of _w
		 */
		void __NORMALIZE__(weight *w_)
		{
			float_t normalizer_ = (float_t)1 ;
			cacu_scalex(w_->s_diff(), w_->count(), normalizer_);
		}



	private:

		blobs* _temp;

	};
}
