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

#ifndef SOLVER_BASE_H_
#define SOLVER_BASE_H_

#include "../math/math_definition.h"
#include "../framework/weight.h"
#include "../framework/network.h"
#include "../framework/model.h"


namespace cacu {

enum train_stage {
	STD_STAGE,
	PRUNE_STAGE,
	UPGRADE_STAGE
};

class solver_base {

public:

	solver_base(network *&net_);

	solver_base(model *&model_);

	virtual ~solver_base();

	inline void set_weight_decay(float_t weight_decay_) {
		_global_weight_decay = weight_decay_;
		_pruneable_decay = _global_weight_decay;
	}

	inline void set_lr(float_t lr_) {
		_global_lr = lr_;
	}

	inline void set_regularize(regularize_type type_) {
		_regularize = type_;
	}

	inline float_t weight_decay() const {
		return _global_weight_decay;
	}

	inline float_t lr() const {
		return _global_lr;
	}

	inline regularize_type regularize() const {
		return _regularize;
	}

	inline void set_gamma(float_t gamma_){
		_gamma = gamma_;
	}

	inline float_t gamma() const {return _gamma;}

	virtual void load_param(const chars_t& config_) = 0;

	/*
	 * applicate to the weights update
	 */
	void updates(int step_);

	void crop_grad(blob* g_);

	void train_iter(int step_);

	/**
	 * change global_lr by rate after several training iterations
	 *
	 */
	inline void set_lr_iter(float_t lr_rate_) {
		CHECK_LT_OP(lr_rate_, 1, "gamma should be less than 1 vs %f", lr_rate_);
		this->_global_lr *= lr_rate_;
		this->_pruneable_lr *= lr_rate_;
	}

	inline network* net() const {return _net;}

	inline void update_direction(grad_direction_type direction_) {
		switch(direction_){
		case maximize:
			_direction = -1.0;
			break;
		case minimize:
			_direction = 1.0;
			break;
		default:
			_direction = 1.0;
			break;
		}

	}

	void positive_regularize(weight *&w_, int weight_index_);

	void cf_regularize(weight *&w_, int weight_index_);

	void bn_regularize(weight *&w_, int weight_index_, vec_i *& upgrade_index_);

	void bn_regularize_storage(blob *blob_, vec_i *& upgrade_index_);

	void upgrade_cf_optimization(weight *&w_, int weight_index_);

	void upgrade_bn_optimization(weight *&w_, int weight_index_, vec_i *& upgrade_index_);

	void reinitialize();

	void reinitial_weight(weight *&w_, int weight_index_);

	void reinitial_bn(weight *&w_, int weight_index_, vec_i *& upgrade_index_);

	inline void set_pruneable_lr(float_t lr_) {
		_pruneable_lr = lr_;
	}

	inline void set_is_pruning(bool is_pruning_){
		_is_pruning = is_pruning_;
	}

	inline void set_is_upgrade_optimize(bool is_upgrade_optimize_){
		_is_upgrade_optimize = is_upgrade_optimize_;
	}

	inline void set_pruneable_alpha(float_t pruneable_alpha_){
		_pruneable_alpha = pruneable_alpha_;
	}

	inline void set_pruneable_decay(float_t pruneable_decay_){
		_pruneable_decay = pruneable_decay_;
	}

	inline void set_train_stage(train_stage stage_){
		_current_stage = stage_;
		stage_config();
	}

	inline void set_positive_regularize(bool is_positive_regularize_){
		_is_positive_regularize = is_positive_regularize_;
		if(_is_positive_regularize)
			LOG_INFO("Use positive regularization!");
	}

	void statement();

	inline void data_recode(void pfunc(weight*&, std::ostream&), std::ostream &os, int op_index_) {
		for(unsigned int i = 0 ; i < _net->get_op(op_index_)->get_weights()->size(); ++i)
			pfunc(_net->get_op(op_index_)->get_weight(i), os);
	}

protected:

	float_t _global_lr = 1.0;

	float_t _global_weight_decay = 0.004;

	float_t _gamma = 0.9;

	regularize_type _regularize = L2;

	network *_net;

	float_t _direction = 1.0;

	float_t _pruneable_lr = 1.0;

	/**
	 * add regular to gradient
	 * where i is the index of _w
	 */
	void __REGULARIZE__(weight *&w_, int weight_index_);

	/**
	 * normalize gradient
	 * where i is the index of _w
	 */
	void __NORMALIZE__(weight *&w_);

	/*
	 * where weight_index denote the weight's id in sovler's vector
	 */
	virtual void update_weight(weight *&w_, int weight_index_, int step_) = 0;

private:

	blobs* _temp;

	int _batch_size;

	bool _is_pruning = false;

	bool _is_upgrade_optimize = false;

	bool _is_positive_regularize = false;

	float_t _pruneable_alpha = 0.9;

	float_t _pruneable_decay = _global_weight_decay;

	train_stage _current_stage = STD_STAGE;

	void stage_config();

};
}


#endif
