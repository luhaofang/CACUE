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

#include "sgd_solver.h"

#include "../../tools/string_utils.h"
#include "../../tools/vec_utils.h"

namespace cacu {

sgd_solver::sgd_solver(network *&net_) :
		solver_base(net_) {

	_momentum = 0.9f;
	_history_v = cacu_allocator::create_blobs();
	for (int i = 0; i < _net->op_count(); ++i) {
		operator_base* op_ = _net->get_op(i);
		for (int j = 0; j < op_->weights_size(); ++j) {
			if (op_->get_weight(j)->variable()){
				blob *history_w = op_->get_weight(j)->copy_create(test, 0);
				_history_v->push_back(history_w);
			}
		}
	}
}

sgd_solver::~sgd_solver() {

	delete _history_v;

}

/**
 * update weight value
 * where weight_index_ is the weight index in _history_v
 */
void sgd_solver::update_weight(weight *&w_, int weight_index_, int step_) {

	blob* history_ = _history_v->asblob(weight_index_);
	float_t learn_rate_ = w_->lr() * _global_lr;
	float_t weight_decay_ = w_->decay() * _global_weight_decay;
	//history_v update
	cacu_saxpby(w_->s_diff(), (float_t)(1.0) * learn_rate_, history_->s_data(),
		_momentum, w_->count());
	//update to weight
	//fixed regularization and weight decay
	cacu_saxpby(history_->s_data(), (float_t)(-1.0), w_->s_data(), (1.0 - learn_rate_ * weight_decay_), w_->count());

//	for (int i = 0; i < w_->num(); ++i) {
//		if (!FIND_FROM_VEC(*w_->update_index(), i)) {
//			//history_v update
//			cacu_saxpby(w_->p_diff(i), (float_t) (1.0) * learn_rate_,
//					history_->p_data(i), _momentum, w_->length());
//			//update to weight
//			//fixed regularization and weight decay
//			cacu_saxpby(history_->p_data(i), (float_t) (-1.0), w_->p_data(i),
//					(1.0 - learn_rate_ * weight_decay_), w_->length());
//		}
//	}
}

void sgd_solver::load_param(const chars_t& config_) {
	ifstream is;
	is.open(config_, ios::in);
	is.precision(numeric_limits<float>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", config_.c_str());
	string file_ = "";
	vector<string> vec;
	while (getline(is, file_)) {
		vec = split(file_, ":");
		if (vec[0] == "learning_rate")
			this->set_lr(strtof(vec[1].c_str(), NULL));
		if (vec[0] == "weight_decay")
			this->set_weight_decay(strtof(vec[1].c_str(), NULL));
		if (vec[0] == "momentum")
			this->set_momentum(strtof(vec[1].c_str(), NULL));
		if (vec[0] == "gamma")
			this->set_gamma(strtof(vec[1].c_str(), NULL));
	}
	is.close();
}

}
