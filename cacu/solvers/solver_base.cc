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
#include "../../tools/vec_utils.h"

#include "../framework/initialize_functions.h"

namespace cacu {

solver_base::solver_base(network *&net_) {
	if (net_->phase() == test)
		LOG_FATAL("Network is created with a test phase!");
	_net = net_;

	_temp = cacu_allocator::create_blobs();
	for (int i = 0; i < _net->op_count(); ++i) {
		operator_base* op_ = _net->get_op(i);
		for (int j = 0; j < op_->weights_size(); ++j) {
			if (op_->get_weight(j)->variable()) {
				blob *temp_w = op_->get_weight(j)->copy_create(test, 0);
				_temp->push_back(temp_w);
			}
		}
	}

	_batch_size = _net->input_blobs()->at(0)->num();

	_direction = 1.0;
}

solver_base::solver_base(model *&model_) {

	_net = NULL;

	_temp = cacu_allocator::create_blobs();
	for (int j = 0; j < model_->get_params()->size(); ++j) {
		if (model_->get_params()->at(j)->variable()) {
			blob *temp_w = model_->get_params()->at(j)->copy_create(test, 0);
			_temp->push_back(temp_w);
		}
	}

	_batch_size = _net->input_blobs()->at(0)->num();

	_direction = 1.0;
}

void solver_base::statement() {
	chars_t stage;
	switch (_current_stage) {
	case STD_STAGE:
		stage = "std_stage";
		break;
	case PRUNE_STAGE:
		stage = "prune_stage";
		break;
	case UPGRADE_STAGE:
		stage = "upgrade_stage";
		break;
	default:
		break;
	}
	LOG_INFO("current stage: %s, positive_regularization: %d", stage.c_str(),
			_is_positive_regularize);
	if (_current_stage == PRUNE_STAGE)
		LOG_INFO("[prune_alpha: %f, prune_weightdecay: %f, prune_lr: %f]",
				_pruneable_alpha, _pruneable_decay, _pruneable_lr);
	LOG_INFO("[global_weightdecay: %f, global_lr: %f]", _global_weight_decay,
			_global_lr);
}

solver_base::~solver_base() {

	delete _temp;
}

void solver_base::stage_config() {
	switch (_current_stage) {
	case STD_STAGE:
		this->_is_pruning = false;
		this->_is_upgrade_optimize = false;
		LOG_INFO("Switch to standard training stage!");
		break;
	case PRUNE_STAGE:
		this->_is_pruning = true;
		//tune back _pruneable_lr to 1
		this->_pruneable_lr = 1;
		this->_is_upgrade_optimize = false;
		LOG_INFO("Switch to pruning training stage!");
		break;
	case UPGRADE_STAGE:
		this->_is_pruning = false;
		this->_is_upgrade_optimize = true;
		this->reinitialize();
		LOG_INFO("Switch to upgrade training stage!");
		break;
	default:
		LOG_FATAL("Wrong type of the training stage!")
		;
	}
	statement();
}

void solver_base::crop_grad(blob* g_) {
}

void solver_base::updates(int step_) {
	int weight_index_ = 0;
	//update weights
	for (int i = 0; i < _net->op_count(); ++i) {
		operator_base* op_ = _net->get_op(i);
		for (int j = 0; j < op_->weights_size(); ++j) {
			//whether the weight is variable
			if (op_->get_weight(j)->variable()) {
				//whether need to update
				if (op_->get_weight(j)->update()) {
					//fix direction of the gradient
					if (_direction != 1.0)
						cacu_scalex(op_->get_weight(j)->s_diff(),
								op_->get_weight(j)->count(), _direction);
					//normalization
					__NORMALIZE__(op_->get_weight(j));
					//add regularization
					__REGULARIZE__(op_->get_weight(j), weight_index_);

					if (_is_pruning) {
						if (op_->get_optype() == CACU_INNERPRODUCT
								|| op_->get_basetype() == CONV_BASE)
							cf_regularize(op_->get_weight(j), weight_index_);
						if (op_->get_optype() == CACU_BATCH_NORMALIZE
								&& (_net->get_op(i - 1)->get_optype()
										== CACU_INNERPRODUCT
										|| op_->get_basetype() == CONV_BASE)) {
							bn_regularize(op_->get_weight(j), weight_index_,
									_net->get_op(i - 1)->get_weight(0)->upgrade_index());
							//reinitialize history_mean & history_var
							bn_regularize_storage(op_->get_storage_blob(j + 9),
									_net->get_op(i - 1)->get_weight(0)->upgrade_index());
						}
					}
					if (_is_upgrade_optimize) {
						if (op_->get_optype() == CACU_INNERPRODUCT
								|| op_->get_basetype() == CONV_BASE)
							upgrade_cf_optimization(op_->get_weight(j),
									weight_index_);
						if (op_->get_optype() == CACU_BATCH_NORMALIZE
								&& (_net->get_op(i - 1)->get_optype()
										== CACU_INNERPRODUCT
										|| op_->get_basetype() == CONV_BASE))
							upgrade_bn_optimization(op_->get_weight(j),
									weight_index_,
									_net->get_op(i - 1)->get_weight(0)->upgrade_index());
					}
					update_weight(op_->get_weight(j), weight_index_, step_);
				}
				//reset weight's gradient
				op_->get_weight(j)->_RESET_DIFF();
				weight_index_++;
			}
		}
//		if(op_->get_basetype() == BN_BASE)
//			_net->get_op<bn_base_op>(i)->update_history();
	}
//	_net->input_blobs()->asblob(0)->_RESET_DIFF();
}

void solver_base::reinitialize() {
	LOG_INFO("Reinitialize silence parameters!");
	int weight_index_ = 0;
	//update weights
	for (int i = 0; i < _net->op_count(); ++i) {
		operator_base* op_ = _net->get_op(i);
		for (int j = 0; j < op_->weights_size(); ++j) {
			//whether the weight is variable
			if (op_->get_weight(j)->variable()) {
				//whether need to update
				if (op_->get_weight(j)->update()) {
					//fix direction of the gradient
					if (op_->get_optype() == CACU_INNERPRODUCT
							|| op_->get_optype() == CACU_CONVOLUTION)
						reinitial_weight(op_->get_weight(j), weight_index_);
					if (op_->get_optype() == CACU_BATCH_NORMALIZE
							&& (_net->get_op(i - 1)->get_optype()
									== CACU_INNERPRODUCT
									|| _net->get_op(i - 1)->get_optype()
											== CACU_CONVOLUTION)) {
						reinitial_bn(op_->get_weight(j), weight_index_,
								_net->get_op(i - 1)->get_weight(0)->upgrade_index());
						//reinitialize history_mean & history_var
						bn_regularize_storage(op_->get_storage_blob(j + 9),
								_net->get_op(i - 1)->get_weight(0)->upgrade_index());
					}
				}
				weight_index_++;
			}
		}
	}
}

void solver_base::train_iter(int step_) {
	_net->set_phase(train);

	_net->forward_propagate();

	_net->backward_propagate();

	this->updates(step_);
}

/**
 * add regular to gradient
 * where i is the index of _w
 */
void solver_base::__REGULARIZE__(weight *&w_, int weight_index_) {
	float_t weight_decay_ = w_->decay() * _global_weight_decay;
	blob *temp_w = _temp->asblob(weight_index_);
	switch (_regularize) {
	case L1:
		cacu_abs_grad(w_->s_data(), temp_w->s_data(), w_->count());
		cacu_saxpy(temp_w->s_data(), weight_decay_, w_->s_diff(), w_->count());
		break;
	case L2:
		cacu_saxpy(w_->s_data(), weight_decay_, w_->s_diff(), w_->count());
		break;
	case NO_REGULARIZATION:
		break;
	default:
		break;
	}
	if (_is_positive_regularize)
		positive_regularize(w_, weight_index_);
}

/**
 * normalize gradient
 * where i is the index of _w
 */
void solver_base::__NORMALIZE__(weight *&w_) {
	float_t normalizer_ = (float_t) (1);
	cacu_scalex(w_->s_diff(), w_->count(), normalizer_);
}

/*
 * following is the prune methods
 * upgrade_cf_optimization : upgrade training for convolution and full connection op.
 * upgrade_bn_optimization : upgrade training for bn op
 * cf_regularize : pruneable training for convolution and full connection op.
 * bn_regularize : pruneable training for bn op
 */

void solver_base::upgrade_cf_optimization(weight *&w_, int weight_index_) {

	for (int i = 0; i < w_->num(); ++i) {
		if (FIND_FROM_VEC(*w_->upgrade_index(), i)) {
			w_->set_pdiff(0, i);
//			cacu_scalex(w_->p_diff(i), w_->length(), 0.1);
		}
	}
}

void solver_base::upgrade_bn_optimization(weight *&w_, int weight_index_,
		vec_i *& upgrade_index_) {

	w_->upgrade_index()->clear();
	for (int i = 0; i < w_->num(); ++i) {
		if (FIND_FROM_VEC(*upgrade_index_, i)) {
			w_->set_pdiff(0, i);
			w_->upgrade_index()->push_back(i);
		}
	}
}

void solver_base::bn_regularize(weight *&w_, int weight_index_,
		vec_i *& upgrade_index_) {

	for (int i = 0; i < w_->num(); ++i) {
		if (!FIND_FROM_VEC(*upgrade_index_, i)) {
			w_->set_pdata(0, i);
		}
	}
}

void solver_base::reinitial_bn(weight *&w_, int weight_index_,
		vec_i *& upgrade_index_) {
	for (int i = 0; i < w_->num(); ++i)
		if (!FIND_FROM_VEC(*upgrade_index_, i)) {
			if (w_->name() == "scale")
				w_->set_pdata(1, i);
			else if (w_->name() == "shift")
				w_->set_pdata(0, i);
		}
}

void solver_base::reinitial_weight(weight *&w_, int weight_index_) {
	if (!_is_upgrade_optimize)
		return;
	if (w_->length() == 1 || weight_index_ == _temp->size() - 2) //|| weight_index_ != 0
		return;
	vec_t temp(w_->length());
	vec_t wtemp(w_->count());
	int index_;

	float wm = 0, ws = 0;
#if __USE_DEVICE__ == ON
	cacu_sumbysize_cpu(BYWIDTH, w_->s_data_cpu(), w_->count(), (float_t) 1, &wm,
			(float_t) 0, w_->count());
#else
	cacu_sumbysize(BYWIDTH, w_->s_data(), w_->count(), (float_t)1, &wm, (float_t)0, w_->count());
#endif
	wm = wm / w_->count();
#if __USE_DEVICE__ == ON
	cacu_sdxsize_cpu(w_->s_data_cpu(false), w_->count(), (float_t) -1 * wm,
			(float_t) 1, &wtemp[0]);
#else
	cacu_sdxsize(w_->s_data(), w_->count(), (float_t)-1 * wm, (float_t)1, &wtemp[0]);
#endif
	cacu_sqr_cpu(&wtemp[0], w_->count(), &wtemp[0]);
	cacu_sumbysize_cpu(BYWIDTH, &wtemp[0], w_->count(), (float_t) 1, &ws,
			(float_t) 0, w_->count());
	ws = sqrtf(ws / (w_->count() - 1));

	//reinitialize silent weight
	for (int i = 0; i < w_->num(); ++i) {
		float_t m = 0, s = 0;
#if __USE_DEVICE__ == ON
		cacu_sumbysize_cpu(BYWIDTH, w_->p_data_cpu(i), w_->length(),
				(float_t) 1, &m, (float_t) 0, w_->length());
#else
		cacu_sumbysize(BYWIDTH, w_->p_data(i), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
#endif
		m = m / w_->length();
#if __USE_DEVICE__ == ON
		cacu_sdxsize_cpu(w_->p_data_cpu(i, false), w_->length(),
				(float_t) -1 * m, (float_t) 1, &temp[0]);
#else
		cacu_sdxsize(w_->p_data(i), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
#endif
		cacu_sqr_cpu(&temp[0], w_->length(), &temp[0]);
		cacu_sumbysize_cpu(BYWIDTH, &temp[0], w_->length(), (float_t) 1, &s,
				(float_t) 0, w_->length());
		s = sqrtf(s / (w_->length() - 1));

		if (s < ws / w_->length()) {
			index_ = FIND_INDEX_VEC(*w_->upgrade_index(), i);
			if (index_ >= 0)
				w_->upgrade_index()->erase(
						w_->upgrade_index()->begin() + index_);
			gaussian_init(w_->p_data(i), w_->length(), wm, ws);
//			w_->set_pdata(0, i);
		} else {
			if (!FIND_FROM_VEC(*w_->upgrade_index(), i))
				w_->upgrade_index()->push_back(i);
		}
	}
}

void solver_base::bn_regularize_storage(blob *blob_, vec_i *& upgrade_index_) {
	for (int i = 0; i < blob_->num(); ++i) {
		//upgrade training
		if (!FIND_FROM_VEC(*upgrade_index_, i)) {
			blob_->set_pdata(0, i);
			//cacu_set_value(blob_->p_data(i), 0, 1);
		}
	}
}

void solver_base::cf_regularize(weight *&w_, int weight_index_) {
	if (w_->length() == 1 || weight_index_ == _temp->size() - 2) //|| weight_index_ != 0
		return;
	vec_t temp(w_->length());
	vec_t wtemp(w_->count());
	blob *temp_w = _temp->asblob(weight_index_);
	temp_w->set_data(1);
	float_t rand_t;
	float_t weight_decay_;

	float wm = 0, ws = 0;
#if __USE_DEVICE__ == ON
	cacu_sumbysize_cpu(BYWIDTH, w_->s_data_cpu(), w_->count(), (float_t) 1, &wm,
			(float_t) 0, w_->count());
#else
	cacu_sumbysize(BYWIDTH, w_->s_data(), w_->count(), (float_t)1, &wm, (float_t)0, w_->count());
#endif
	wm = wm / w_->count();
#if __USE_DEVICE__ == ON
	cacu_sdxsize_cpu(w_->s_data_cpu(false), w_->count(), (float_t) -1 * wm,
			(float_t) 1, &wtemp[0]);
#else
	cacu_sdxsize(w_->s_data(), w_->count(), (float_t)-1 * wm, (float_t)1, &wtemp[0]);
#endif
	cacu_sqr_cpu(&wtemp[0], w_->count(), &wtemp[0]);
	cacu_sumbysize_cpu(BYWIDTH, &wtemp[0], w_->count(), (float_t) 1, &ws,
			(float_t) 0, w_->count());
	ws = sqrtf(ws / (w_->count() - 1));
	int count = 0, s_count = 0;
	float_t lambda;
	int index_;

	weight_decay_ = w_->decay() * _pruneable_decay / _pruneable_lr; // * ws; * _pruneable_lr

	//add regularization
	//cacu_sdxsize(w_->s_data(), w_->count(), -wm,  -ws * w_->count() / weight_decay_, w_->s_diff());

	for (int i = 0; i < w_->num(); ++i) {
		float m = 0, s = 0;
#if __USE_DEVICE__ == ON
		cacu_sumbysize_cpu(BYWIDTH, w_->p_data_cpu(i, false), w_->length(),
				(float_t) 1, &m, (float_t) 0, w_->length());
#else
		cacu_sumbysize(BYWIDTH, w_->p_data(i), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
#endif
		m = m / w_->length();
#if __USE_DEVICE__ == ON
		cacu_sdxsize_cpu(w_->p_data_cpu(i, false), w_->length(),
				(float_t) -1 * m, (float_t) 1, &temp[0]);
#else
		cacu_sdxsize(w_->p_data(i), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
#endif
		cacu_sqr_cpu(&temp[0], w_->length(), &temp[0]);
		cacu_sumbysize_cpu(BYWIDTH, &temp[0], w_->length(), (float_t) 1, &s,
				(float_t) 0, w_->length());
		s = sqrtf(s / (w_->length() - 1));

		rand_t = min(s / ws, (float_t) 1);

		lambda =
				(rand_t >= _pruneable_alpha) ?
						0 : ((_pruneable_alpha - rand_t) / _pruneable_alpha);

		cacu_scalex(w_->p_diff(i), w_->length(),
				min(rand_t / _pruneable_alpha, (float_t) 1));
		cacu_abs_grad(w_->p_data(i), temp_w->p_data(i), w_->length());
		cacu_saxpy(temp_w->p_data(i), weight_decay_ * lambda, w_->p_diff(i),
				w_->length());

		if (s >= ws / w_->length() && rand_t < _pruneable_alpha)
			s_count += 1;

		if (s < ws / w_->length()) {
			count += 1;
			index_ = FIND_INDEX_VEC(*w_->upgrade_index(), i);
			if (index_ >= 0)
				w_->upgrade_index()->erase(
						w_->upgrade_index()->begin() + index_);
		} else {
			if (!FIND_FROM_VEC(*w_->upgrade_index(), i))
				w_->upgrade_index()->push_back(i);
		}
		if (s >= ws / w_->length() && rand_t < _pruneable_alpha)
			LOG_DEBUG("%.2f [%.8f, %.8f, %.8f, %.8f, %.8f]", _pruneable_alpha,
					m, s, wm, ws, s / ws);
	}
	LOG_DEBUG("[%d | %d | %d] [%.8f, %.8f]", count, s_count, w_->num(), wm, ws);

}

//for using to increase the model's generalization
void solver_base::positive_regularize(weight *&w_, int weight_index_) {
	if (w_->length() == 1 || weight_index_ == _temp->size() - 2)
		return;
	vec_t temp(w_->length());
	vec_t residual(w_->length());
	vec_t wtemp(w_->count());
	vec_t wresidual(w_->count());

	w_->update_index()->clear();
	int count = 0;
	float_t acc_cdf = 0;
	float_t cdf = 0;
	float_t rand;
	float_t rate;
	int sign = 0;

	float wm = 0, ws = 0;
	cacu_copy(w_->s_diff(), w_->length(), &wresidual[0]);
	cacu_saxpby(w_->s_data(), 1., &wresidual[0], -1., w_->count());
	cacu_sumbysize_cpu(BYWIDTH, &wresidual[0], w_->count(), (float_t) 1, &wm,
			(float_t) 0, w_->count());
	wm = wm / w_->count();
	cacu_sdxsize(&wresidual[0], w_->count(), (float_t) -1 * wm, (float_t) 1,
			&wtemp[0]);
	cacu_sqr_cpu(&wtemp[0], w_->count(), &wtemp[0]);
	cacu_sumbysize_cpu(BYWIDTH, &wtemp[0], w_->count(), (float_t) 1, &ws,
			(float_t) 0, w_->count());
	ws = sqrtf(ws / (w_->count() - 1));

	for (int i = 0; i < w_->num(); ++i) {
		float_t m = 0, s = 0;
		float_t md = 0, sd = 0;
#if __USE_DEVICE__ == ON
		cacu_sumbysize_cpu(BYWIDTH, w_->p_data_cpu(i, false), w_->length(),
				(float_t) 1, &m, (float_t) 0, w_->length());
		cacu_sumbysize_cpu(BYWIDTH, w_->p_diff_cpu(i), w_->length(),
				(float_t) 1, &md, (float_t) 0, w_->length());
#else
		cacu_sumbysize(BYWIDTH, w_->p_data(i), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
		cacu_sumbysize(BYWIDTH, w_->p_diff(i), w_->length(), (float_t)1, &md, (float_t)0, w_->length());
#endif
		m = m / w_->length();
		md = md / w_->length();
#if __USE_DEVICE__ == ON
		cacu_sdxsize_cpu(w_->p_data_cpu(i, false), w_->length(),
				(float_t) -1 * m, (float_t) 1, &temp[0]);
#else
		cacu_sdxsize(w_->p_data(i), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
#endif
		cacu_sqr_cpu(&temp[0], w_->length(), &temp[0]);
		cacu_sumbysize_cpu(BYWIDTH, &temp[0], w_->length(), (float_t) 1, &s,
				(float_t) 0, w_->length());
		s = sqrtf(s / (w_->length() - 1));

#if __USE_DEVICE__ == ON
		cacu_sdxsize_cpu(w_->p_diff_cpu(i, false), w_->length(),
				(float_t) -1 * md, (float_t) 1, &temp[0]);
#else
		cacu_sdxsize(w_->p_diff(i), w_->length(), (float_t)-1 * md, (float_t)1, &temp[0]);
#endif
		cacu_sqr_cpu(&temp[0], w_->length(), &temp[0]);
		cacu_sumbysize_cpu(BYWIDTH, &temp[0], w_->length(), (float_t) 1, &sd,
				(float_t) 0, w_->length());
		sd = sqrtf(sd / (w_->length() - 1));

		rate = sd;	// / w_->count();

		cdf = (md >= 0) ?
				(1. - cacu_normal_cdf(0, md, rate)) :
				(cacu_normal_cdf(0, md, rate));

		sign = 1;
		rand = urand(0, 1);
		if (rand >= cdf) {
			count += 1;
			w_->update_index()->push_back(i);
			sign = 0;
		}
		if (weight_index_ == 0)
			LOG_DEBUG("%d, %f [%.8f, %.8f, %.8f, %.8f, %.8f]", sign, rand, m, s,
					md, sd, cdf);
		acc_cdf += cdf / w_->num();
	}
	LOG_DEBUG("[%d, %.8f, %.8f]", w_->num() - count, wm, ws);
}

}
