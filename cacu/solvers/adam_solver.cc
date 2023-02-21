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

#include <math.h>

#include "../../tools/string_utils.h"

#include "../../tools/vec_utils.h"

namespace cacu {

adam_solver::adam_solver(network *&net_) :
		solver_base(net_) {
	_history_m = cacu_allocator::create_blobs();
	_history_v = cacu_allocator::create_blobs();
	for (int i = 0; i < _net->op_count(); ++i) {
		operator_base* op_ = _net->get_op(i);
		for (int j = 0; j < op_->weights_size(); ++j) {
			if (op_->get_weight(j)->variable()) {
				blob *history_w_m = op_->get_weight(j)->copy_create(train, 0);
				blob *history_w_v = op_->get_weight(j)->copy_create(train, 0);
				_history_m->push_back(history_w_m);
				_history_v->push_back(history_w_v);
			}
		}
	}
}

adam_solver::~adam_solver() {

	delete _history_m;
	delete _history_v;

}

/**
 * update weight value
 * where weight_index_ is the weight index in _history_v
 */
void adam_solver::update_weight(weight *&w_, int weight_index_, int step_) {

	if (step_ == 0)
		LOG_FATAL("adam optimizer must start from iteration 1 vs %d!", step_);

	blob* history_m = _history_m->asblob(weight_index_);
	blob* history_v = _history_v->asblob(weight_index_);
	float_t learn_rate_ = w_->lr() * _global_lr;
	float_t weight_decay_ = w_->decay() * _global_weight_decay;

//
	//history_v update
	cacu_saxpby(w_->s_diff(), (1. - _beta1), history_v->s_data(), _beta1, w_->count());
	cacu_sqr(w_->s_diff(), w_->count(), w_->s_diff());
	cacu_saxpby(w_->s_diff(), (1. - _beta2), history_m->s_data(), _beta2, w_->count());

	cacu_copy(history_v->s_data(), history_v->count(), history_v->s_diff());
	cacu_scalex(history_v->s_diff(), history_v->count(), (1. / (1. - powf(_beta1, step_))));
	cacu_copy(history_m->s_data(), history_m->count(), history_m->s_diff());
	cacu_scalex(history_m->s_diff(), history_m->count(), (1. / (1. - powf(_beta2, step_))));
	cacu_sdxsize(history_m->s_diff(), history_m->count(), _epsilon, 1., history_m->s_diff());
	cacu_root(history_m->s_diff(), history_m->count(), history_m->s_diff());
	cacu_invx(history_m->s_diff(), history_m->count(), w_->s_diff());
	cacu_scalex(history_v->s_diff(), history_v->count(), learn_rate_);
	cacu_ssx(history_v->s_diff(), history_v->count(), w_->s_diff());

	//update to weight
	//fixed regularization and weight decay
	cacu_saxpby(w_->s_diff(), (float_t) (-1.0), w_->s_data(),
			(1.0 - learn_rate_ * weight_decay_), w_->count());

//	for (int i = 0; i < w_->num(); ++i) {
//		if (!FIND_FROM_VEC(*w_->update_index(), i)) {
//			//history_v update
//			cacu_saxpby(w_->p_diff(i), (float_t) (1.0f - _beta1),
//					history_m->p_data(i), _beta1, w_->length());
//			cacu_sqr(w_->p_diff(i), w_->length(), w_->p_diff(i));
//			cacu_saxpby(w_->p_diff(i), (float_t) (1.0f - _beta2),
//					history_v->p_data(i), _beta2, w_->length());
//
//			cacu_copy(history_m->p_data(i), history_m->length(),
//					history_m->p_diff(i));
//			cacu_root(history_v->p_data(i), history_v->length(),
//					history_v->p_diff(i));
//			cacu_sdxsize(history_v->p_diff(i), history_v->length(), _epsilon,
//					(float_t) 1.0, history_v->p_diff(i));
//			cacu_scalex(history_m->p_diff(i), history_m->length(),
//					learn_rate_ * (float_t) (1.0) * correction);
//			cacu_cdxsize(history_m->p_diff(i), history_m->length(),
//					history_v->p_diff(i), history_v->length(), w_->p_diff(i));
//
//			//update to weight
//			//fixed regularization and weight decay
//			cacu_saxpby(w_->p_diff(i), (float_t) (-1.0), w_->p_data(i),
//					(1.0 - learn_rate_ * weight_decay_), w_->length());
//		}
//	}
}

void adam_solver::load_param(const chars_t& config_) {

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
		if (vec[0] == "beta1")
			this->set_beta1(strtof(vec[1].c_str(), NULL));
		if (vec[0] == "beta2")
			this->set_beta2(strtof(vec[1].c_str(), NULL));
		if (vec[0] == "gamma")
			this->set_gamma(strtof(vec[1].c_str(), NULL));
	}
	is.close();
}

}
