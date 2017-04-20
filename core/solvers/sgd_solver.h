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

	class sgd_solver : public solver_base
	{

	public:

		sgd_solver(network *&net_) : solver_base(net_){
			_history_v = cacu_allocator::create_blobs();
			for(int i = 0; i < _net->op_count(); ++i)
			{
				operator_base* op_ = _net->get_op(i);
				for(int j = 0; j < op_->weights_size(); ++j)
				{
					blob *history_w = op_->get_weight(j)->copy_create(test);
					_history_v->push_back(history_w);
				}
			}
		};

		~sgd_solver(){

			delete _history_v;

		};

		/**
		 * update weight value
		 * where weight_index_ is the weight index in _history_v
		 */
		virtual const void update_weight(weight* w_, int weight_index_) override
		{
			//printf("%d:",w_->count());

			//cacu_print(w_->s_data(),1);


			blob* history_ = (blob*)_history_v->at(weight_index_);
			float_t learn_rate_ = w_->lr() * _global_lr;
			//normalization
			__NORMALIZE__(w_);
			//add regular
			__REGULARIZE__(w_ ,weight_index_);
			//history_v update
			cacu_saxpby(w_->s_diff(), learn_rate_, history_->s_data(), _momentum, w_->count());
			//update to weight
			cacu_saxpy_atomic(history_->s_data(), -1, w_->s_data(), w_->count());

			//cacu_print(w_->s_data(),w_->count());
			//cout<<endl<<endl;
			//cacu_print(history_->s_data(),1);
			//cacu_print(w_->s_data(),1);
			//cout << endl;
			//echo();
		}

		inline void set_momentum(float_t momentum_){ _momentum = momentum_ ;}

		inline float_t momentum(){ return _momentum; }

		void echo()
		{

		}



	protected:


	private:

		float_t _momentum = 0.9f;

		blobs* _history_v;

	};
}
