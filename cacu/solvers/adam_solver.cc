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


#include "adam_solver.h"

#include "../../tools/string_utils.h"
#include <math.h>

namespace cacu {

adam_solver::adam_solver(network *&net_) :
		solver_base(net_) {
	_history_s = cacu_allocator::create_blobs();
	_history_r = cacu_allocator::create_blobs();
	for (int i = 0; i < _net->op_count(); ++i) {
		operator_base* op_ = _net->get_op(i);
		for (int j = 0; j < op_->weights_size(); ++j) {
			blob *history_w_s = op_->get_weight(j)->copy_create(train, 0);
			blob *history_w_r = op_->get_weight(j)->copy_create(train, 0);
			_history_s->push_back(history_w_s);
			_history_r->push_back(history_w_r);
		}
	}
}

adam_solver::~adam_solver() {

	delete _history_s;
	delete _history_r;

}

/**
 * update weight value
 * where weight_index_ is the weight index in _history_v
 */
void adam_solver::update_weight(weight* w_, int weight_index_, int step_) {

	if (w_->update()) {
		blob* history_s = (blob*)_history_s->at(weight_index_);
		blob* history_r = (blob*)_history_r->at(weight_index_);
		float_t learn_rate_ = w_->lr() * _global_lr * std::sqrt(1.0 - pow(_beta, step_)) /
			      (1.0 - pow(_alpha, step_));
		//normalization
		__NORMALIZE__(w_);
		//add regular
		__REGULARIZE__(w_, weight_index_);
		//history_v update
		cacu_saxpby(w_->s_diff(), (float_t)(1.0 - _alpha), history_s->s_data(),
				_alpha, w_->count());
		cacu_sqr(w_->s_diff(),w_->count(),w_->s_diff());
		cacu_saxpby(w_->s_diff(), (float_t)(1.0 - _beta), history_r->s_data(),
						_beta, w_->count());
		cacu_copy(history_s->s_data(),history_s->count(), history_s->s_diff());
		cacu_copy(history_r->s_data(),history_r->count(), history_r->s_diff());
		cacu_scalex(history_s->s_diff(), history_s->count(), 1.0 / (1.0 - std::pow(_alpha, step_)));
		cacu_scalex(history_r->s_diff(), history_r->count(), 1.0 / (1.0 - std::pow(_beta, step_)));
		cacu_scalex(history_s->s_diff(), history_s->count(), learn_rate_ * _direction);
		cacu_root(history_r->s_diff(),history_r->count(),history_r->s_diff());
		cacu_sdxsize<float_t>(history_r->s_diff(),history_r->count(), _epsilon, 1.0, history_r->s_diff());
		cacu_cdxsize(history_s->s_diff(), history_s->count(), history_r->s_diff(), history_r->count(), w_->s_diff());

		//update to weight
		cacu_saxpy(w_->s_diff(), (float_t)(1), w_->s_data(), w_->count());
	}
}

void adam_solver::load_param(chars_t config_)
{

	ifstream is = ifstream(config_);
	is.precision(numeric_limits<float>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", config_.c_str());
	string file_ = "";
	vector<string> vec;
	while (getline(is, file_)) {
		vec = split(file_, ":");
		if(vec[0] == "learning_rate")
			this->set_lr(strtof(vec[1].c_str(), NULL));
		if(vec[0] == "weight_decay")
			this->set_weight_decay(strtof(vec[1].c_str(), NULL));
		if(vec[0] == "alpha")
			this->set_alpha(strtof(vec[1].c_str(), NULL));
		if(vec[0] == "beta")
			this->set_beta(strtof(vec[1].c_str(), NULL));
	}
	is.close();
}

}
