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

#include "network.h"

namespace cacu {

network::network(blobs *&datas_) {
	_input_blobs = datas_;

	_ops = new vector<operator_base*>();
	_layers = new vector<layer_base*>();
}

network::~network() {

	delete _input_blobs;
	for (int i = 0; i < _layers->size(); ++i) {
		delete _layers->at(i);
		_layers->at(i) = NULL;
	}
	delete _layers;
	//DON'T to release the op's memory
	delete _ops;
}

/*
 * Class network unlike layer & layer_block,
 * it maintains the abstract layer's operator entity.
 *
 */
network& network::operator <<(layer_block* const &layer_block_) {
	for (int i = 0; i < layer_block_->length(); ++i) {
		_layers->push_back(layer_block_->layer_bases(i));
		for (int j = 0; j < layer_block_->layer_bases(i)->op_count(); ++j)
			_ops->push_back(layer_block_->layer_bases(i)->op(j));
	}
	return *this;
}

network& network::operator <<(layer_block &layer_block_) {
	for (int i = 0; i < layer_block_.length(); ++i) {
		_layers->push_back(layer_block_.layer_bases(i));
		for (int j = 0; j < layer_block_.layer_bases(i)->op_count(); ++j)
			_ops->push_back(layer_block_.layer_bases(i)->op(j));
	}
	return *this;
}

network& network::operator <<(layer_base* const &layer_) {

	_layers->push_back(layer_);
	for (int j = 0; j < layer_->op_count(); ++j)
		_ops->push_back(layer_->op(j));
	return *this;
}

network& network::operator <<(layer_base &layer_) {

	_layers->push_back(&layer_);
	for (int j = 0; j < layer_.op_count(); ++j)
		_ops->push_back(layer_.op(j));
	return *this;
}

void network::predict() {
	if (test != phase()) {
		set_phase(test);
	}
	for (unsigned int i = 0; i < _layers->size(); ++i) {
		_layers->at(i)->operate();
		//LOG_DEBUG("layer: %d",i);
	}
}

void network::forward_propagate() {
	for (unsigned int i = 0; i < _layers->size(); ++i) {
		//LOG_DEBUG("forward layer: %d",i);
		_layers->at(i)->operate();
	}
}

void network::back_propagate() {
	//et_op(0)->__NEED_BACK_PROPAGATE__(false);
	for (int i = _layers->size() - 1; i >= 0; --i) {
		_layers->at(i)->grad();
		//LOG_DEBUG("backward layer: %d",i);
	}
}

void network::set_weights_type(param_init_type type_, float_t value) {
	for (int i = op_count() - 1; i >= 0; --i) {
		for (int j = 0; j < get_op(i)->weights_size(); ++j) {
			get_op(i)->get_weight(j)->set_init_type(type_, value);
		}
	}
}

void network::output_blobs() {
	for (int i = 0; i < op_count(); ++i) {
		blob_base * bp = get_op(i)->out_data<blob_base>();
		if (bp != NULL && bp->is_output())
			LOG_DEBUG("%d", i);
		else if (bp == NULL) {
			for (int j = 0; j < get_op(i)->out_datas()->size(); ++j) {
				if (get_op(i)->out_datas()->at(j)->is_output())
					LOG_DEBUG("%d:%d", i, j);
			}
		}
	}
}

void network::load_weights(chars_t modelpath) {
	std::ifstream is(modelpath, ios::binary);
	is.precision(std::numeric_limits<float_t>::digits10);
	if (!is)
		LOG_FATAL("model file %s cannot be opened!", modelpath.c_str());
	for (int i = 0; i < op_count(); ++i) {
		get_op(i)->load(is);
	}
	is.close();
	LOG_INFO("Initialize model by : %s", modelpath.c_str());
}

void network::load_weights_from(chars_t modelpath, int op_start, int op_end) {
	std::ifstream is(modelpath, ios::binary);
	is.precision(std::numeric_limits<float_t>::digits10);
	if (!is)
		LOG_FATAL("model file %s cannot be opened!", modelpath.c_str());
	CHECK_LE_OP(op_end,op_count(),"end index must less than current op count!");
	CHECK_LT_OP(op_start,op_end,"start index must less than end index!");
	for (int i = op_start; i < op_end; ++i) {
		//LOG_DEBUG("%d", i);
		get_op(i)->load(is);
	}
	is.close();
	LOG_INFO("Initialize model from op[%d] to op[%d] by : %s", op_start, op_end, modelpath.c_str());
}

void network::load_weights_from(std::ifstream &is, int op_start, int op_end) {

	CHECK_LE_OP(op_end,op_count(),"end index must less than current op count!");
	CHECK_LT_OP(op_start,op_end,"start index must less than end index!");
	for (int i = op_start; i < op_end; ++i) {
		get_op(i)->load(is);
	}
	LOG_INFO("Initialize model from op[%d] to op[%d]", op_start, op_end);
}

void network::save_weights(chars_t modelpath) {
	std::ofstream os(modelpath, ios::binary);
	os.precision(std::numeric_limits<float_t>::digits10);
	for (int i = 0; i < op_count(); ++i) {
		get_op(i)->save(os);
	}
	os.close();
	LOG_INFO("The model is saved at : %s", modelpath.c_str());
}

void network::check() {
	for (unsigned int i = 0; i < _ops->size(); ++i) {
		_ops->at(i)->echo();
	}
}

void network::set_phase(phase_type phase_) {
	for (unsigned int i = 0; i < _ops->size(); ++i) {
		_ops->at(i)->set_phase(phase_);
	}
}

void network::set_update_weight(bool isupdate_)
{
	for (unsigned int i = 0; i < _ops->size(); ++i) {
		_ops->at(i)->set_is_update_weight(isupdate_);
		if(_ops->at(i)->_TYPE()==CACU_BATCH_NORMALIZE)
		{
			((batch_normalize_op *)_ops->at(i))->set_is_use_global_stats(isupdate_==false);
		}
	}
}



}
