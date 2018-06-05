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

#include "solver_base.h"

#include "../math/math_functions.h"

namespace cacu {

solver_base::solver_base(network *&net_) {
	_net = net_;

	_temp = cacu_allocator::create_blobs();
	for (int i = 0; i < _net->op_count(); ++i) {
		operator_base* op_ = _net->get_op(i);
		for (int j = 0; j < op_->weights_size(); ++j) {
			blob *temp_w = op_->get_weight(j)->copy_create(test,0);
			_temp->push_back(temp_w);
		}
	}

	_batch_size = _net->input_blobs()->at(0)->num();
}

solver_base::~solver_base() {

	delete _temp;
}

void solver_base::crop_grad(blob* g_) {
}

void solver_base::updates(int step_){
	int weight_index_ = 0;
	//update weights
	for (int i = 0; i < _net->op_count(); ++i) {
		operator_base* op_ = _net->get_op(i);
		for (int j = 0; j < op_->weights_size(); ++j) {
			update_weight(op_->get_weight(j), weight_index_, step_);
			weight_index_++;
		}
	}
}

void solver_base::train_iter(int step_) {
	_net->set_phase(train);

	_net->forward_propagate();

	_net->back_propagate();

	this->updates(step_);
}

/**
 * add regular to gradient
 * where i is the index of _w
 */
void solver_base::__REGULARIZE__(weight *w_, int weight_index_) {
	float_t weight_decay_ = w_->decay() * _global_weight_decay;
	blob* temp = (blob*) _temp->at(weight_index_);
	switch (_regularize) {
	case L1:
		//temp->set_value(1);
		cacu_sdxsize<float_t>(w_->s_diff(), w_->count(), weight_decay_, 1.0, w_->s_diff());
		//cacu_saxpy(temp->s_data(), weight_decay_, w_->s_diff(), w_->count());
		break;
	case L2:
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
void solver_base::__NORMALIZE__(weight *w_) {
	float_t normalizer_ = (float_t) (1);
	cacu_scalex(w_->s_diff(), w_->count(), normalizer_);
}

}
