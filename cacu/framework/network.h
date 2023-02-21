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

#ifndef NETWORK_H_
#define NETWORK_H_

#include "layer.h"
#include "layer_block.h"
#include "graphic_definition.h"

namespace cacu {

class network {

public:
	network();

	network(blobs *&datas_);

	~network();

	/*
	 * Class network unlike layer & layer_block,
	 * it maintains the abstract layer's operator entity.
	 *
	 */
	network& operator <<(layer_block* layer_block_);

	network& operator <<(layer_block& layer_block_);

	network& operator <<(layer* const &layer_);

	network& operator <<(layer& layer_);

	network& operator <<(operator_base*& op_);

	inline layer *layers(int i) {
		assert(i >= 0);
		return dynamic_cast<layer*>(_layers->at(i));
	}

	inline layer *layer_bases(int i) {
		assert(i >= 0);
		return _layers->at(i);
	}

	inline int layer_count() const {
		assert(_layers->size() < _MAX_INT_);
		return _layers->size();
	}

	void predict();

	void forward_propagate(int op_num = 0);

	void forward_to(int op_num);

	void backward_propagate(int op_num = -1);

	void backward_to(int op_num);

	void set_weights_type(param_init_type type_, float_t value = 0);

	inline void set_weight(int op_id, param_init_type type_, float_t value) {
		get_op(op_id)->get_weight(0)->set_init_type(type_, value);
	}

	inline operator_base *get_op(int i) const {
		assert(i >= 0);
		return _ops->at(i);
	}

	/*
	 * This function will be removed in the next version.
	 * Duplicated method!
	 */
	template<typename OPTYPE>
	inline OPTYPE *get_op(int i, op_name optype_) const {
		if (optype_ == _ops->at(i)->_TYPE()) {
			OPTYPE* op = dynamic_cast<OPTYPE*>(_ops->at(i));
			if (op == NULL)
				LOG_FATAL("op type casting failed!");
			return op;
		} else {
			LOG_FATAL(
					"Shot! You are using a wrong type operator casting as %s!",
					MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()).c_str());
		}
	}

	template<typename OPTYPE>
	inline OPTYPE *get_op(int i) const {
		if (MACRO_FACTORY_OP::get_op_type(
		MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()))
				== _ops->at(i)->_TYPE()) {
			OPTYPE* op = dynamic_cast<OPTYPE*>(_ops->at(i));
			if (op == NULL)
				LOG_FATAL("op type casting failed!");
			return op;
		} else {
			LOG_FATAL(
					"Shot! You are using a wrong type operator casting as %s!",
					MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()).c_str());
		}
	}

	template<typename OPTYPE>
	inline vector<OPTYPE*> get_ops() const {
		vector<OPTYPE*> ops;
		for (unsigned int i = 0; i < _ops->size(); ++i) {
			if (_ops->at(i)->_TYPE() == MACRO_FACTORY_OP::get_op_type(
			MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name())))
				ops.push_back(dynamic_cast<OPTYPE*>(_ops->at(i)));
		}
		if (ops.size() == 0)
			LOG_WARNING("There is no operator be found as %s",
					MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()).c_str());
		return ops;
	}

	template<typename OPTYPE>
	inline vector<OPTYPE*> get_ops(const chars_t& custom_op_name_) const {
		vector<OPTYPE*> ops;
		for (unsigned int i = 0; i < _ops->size(); ++i) {
			if (_ops->at(i)->_TYPE() == MACRO_FACTORY_OP::get_op_type(
			MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()))
					&& _ops->at(i)->get_op_name().c_str() == custom_op_name_)
				ops.push_back(dynamic_cast<OPTYPE*>(_ops->at(i)));
		}
		if (ops.size() == 0)
			LOG_WARNING("There is no operator be found as %s, with name of %s",
					MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()).c_str(),
					custom_op_name_.c_str());
		return ops;
	}

	inline int get_op_index(operator_base *op_) {
		int index = -1;
		for (unsigned int i = 0; i < _ops->size(); ++i) {
			if (op_ == _ops->at(i)) {
				index = i;
				break;
			}
		}
		CHECK_GE_OP(index, 0, "Can not identify operator!");
		return index;
	}

	inline int op_count() const {
		assert(_ops->size() < _MAX_INT_);
		return _ops->size();
	}

	inline blobs *input_blobs() {
		return _input_blobs;
	}

	inline blob *output_blob() {
		if(_ops->size() == 0 && _input_blobs != NULL)
			return _input_blobs->asblob(0);
		return _ops->at(_ops->size() - 1)->out_data<blob>();
	}

	void output_blobs();

	inline phase_type phase() {
		return _input_blobs->at(0)->phase();
	}

	void _INIT_NETWORK() {
		for (unsigned int i = 0; i < _ops->size(); ++i) {
			_ops->at(i)->check();
			_ops->at(i)->initial();
			_ops->at(i)->init_weights();
		}
	}

	void load_weights(const chars_t& modelpath) {
		std::ifstream is(modelpath, ios::binary);
		is.precision(std::numeric_limits<float_t>::digits10);
		if (!is)
			LOG_FATAL("model file %s cannot be opened!", modelpath.c_str());
		for (int i = 0; i < op_count(); ++i) {
			//LOG_DEBUG("%d: %d", i, get_op(i)->get_optype());
			get_op(i)->load(is);
		}
		is.close();
		LOG_INFO("Initialize model by : %s", modelpath.c_str());
	}

	void load_weights_reverse(const chars_t& modelpath) {
		std::ifstream is(modelpath, ios::binary);
		is.precision(std::numeric_limits<float_t>::digits10);
		if (!is)
			LOG_FATAL("model file %s cannot be opened!", modelpath.c_str());
		for (int i = op_count() - 1; i >= 0; --i) {
			//LOG_DEBUG("%d: %d", i, get_op(i)->get_optype());
			get_op(i)->load(is);
		}
		is.close();
		LOG_INFO("Initialize model by : %s", modelpath.c_str());
	}

	void load_weights_from(const chars_t& modelpath, int op_start, int op_end) {
		std::ifstream is(modelpath, ios::binary);
		is.precision(std::numeric_limits<float_t>::digits10);
		if (!is)
			LOG_FATAL("model file %s cannot be opened!", modelpath.c_str());
		CHECK_LE_OP(op_end, op_count(),
				"end index must less than current op count!");
		CHECK_LT_OP(op_start, op_end, "start index must less than end index!");
		for (int i = op_start; i < op_end; ++i) {
			//LOG_DEBUG("%d", i);
			get_op(i)->load(is);
		}
		is.close();
		LOG_INFO("Initialize model from op[%d] to op[%d] by : %s", op_start,
				op_end, modelpath.c_str());
	}

	void load_weights_from(std::ifstream &is, int op_start, int op_end);

	void save_weights(const chars_t& modelpath) {
		std::ofstream os(modelpath, ios::binary);
		os.precision(std::numeric_limits<float_t>::digits10);
		for (int i = 0; i < op_count(); ++i) {
			get_op(i)->save(os);
		}
		os.close();
		LOG_INFO("The model is saved at : %s", modelpath.c_str());
	}

	void check();

	void set_phase(phase_type phase_);

	void set_update_weight(bool isupdate_);

	inline void _RESET_WEIGHT_DIFF() {
		for (int i = op_count() - 1; i >= 0; --i) {
			for (int j = 0; j < get_op(i)->weights_size(); ++j) {
				get_op(i)->get_weight(j)->_RESET_DIFF();
			}
		}
	}

	inline void _RESET_DATA() {
		for (int i = op_count() - 1; i >= 0; --i) {
			get_op(i)->LOOP_INIT_DATA_();
		}
	}

	inline void _RESET_DIFF() {
		for (int i = op_count() - 1; i >= 0; --i) {
			get_op(i)->LOOP_INIT_DIFF_();
		}
	}

	inline void set_conditional_bn_morphing(int step, int cls1, int cls2) {
		for (unsigned int i = 0; i < _ops->size(); ++i) {
			if (_ops->at(i)->_TYPE() == CACU_CONDITIONAL_BN)
				dynamic_cast<conditional_bn_op *>(_ops->at(i))->morphing_step(
						step, cls1, cls2);
		}
	}

	inline void set_conditional_bn_morphing_interpolated_shifting_rate(
			float_t interpolated_shifting_rate_) {
		for (unsigned int i = 0; i < _ops->size(); ++i) {
			if (_ops->at(i)->_TYPE() == CACU_CONDITIONAL_BN)
				dynamic_cast<conditional_bn_op *>(_ops->at(i))->set_interpolated_shifting_rate(
						interpolated_shifting_rate_);
		}
	}

	inline void delete_op(int op_id) {
		if (op_id < 0 || op_id >= op_count())
			LOG_FATAL("Out of op count length %d vs %d!", op_id, op_count());
		LOG_INFO("Delete op %s from network!",
				MACRO_FACTORY_OP::get_op_name(_ops->at(op_id)->get_optype()).c_str());
		_ops->erase(_ops->begin() + op_id);
	}

	/*
	 * This is a test function for clip negative weight
	 */

	inline void weights_pre_proc(void pfunc(weight*&)) {
		for (int i = op_count() - 1; i >= 0; --i) {
			for (int j = 0; j < get_op(i)->weights_size(); ++j) {
				if (get_op(i)->get_optype() == CACU_INNERPRODUCT
						|| get_op(i)->get_optype() == CACU_CONVOLUTION)
					get_op(i)->weight_pre_proc(pfunc, get_op(i)->get_weight(j));
			}
		}
	}

	template<typename OPTYPE>
	inline void set_operator_statement(void pfunc(operator_base*&)) {
		vector<OPTYPE *> ops_ = this->get_ops<OPTYPE>();
		for (int i = 0; i < ops_.size(); ++i)
			pfunc((operator_base*&) ops_[i]);
	}

	void serialize_model(const chars_t& modelpath);

	void network_pruning(const chars_t& model_path);

	inline void set_is_use_bias(bool is_use_bias) {
		vector<convolution_op*> convs = get_ops<convolution_op>();
		for (int i = 0; i < convs.size(); ++i)
			convs[i]->set_is_use_bias(is_use_bias);

		vector<inner_product_op*> ips = get_ops<inner_product_op>();
		for (int i = 0; i < ips.size(); ++i)
			ips[i]->set_is_use_bias(is_use_bias);
	}

	inline void push_back_op(operator_base* op_) {
		_ops->push_back(op_);
		op_->echo();
		_ASSOCIATE_INPUT_DATA();
	}

	void upgrade_bn_initial();

private:

	vector<layer*> *_layers;

	vector<operator_base*> *_ops;

	blobs* _input_blobs = NULL;

	void _ASSOCIATE_INPUT_DATA();

	vec_i _weight_clip(weight *&w_);

	vector<graphic_edge> _phrase_tensor_path();

};

}

#endif
