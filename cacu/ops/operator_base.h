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

#ifndef OPERATOR_BASE_H_
#define OPERATOR_BASE_H_

#include "ops_definition.h"

#include "../framework/blob.h"
#include "../framework/bin_blob.h"
#include "../framework/blobs.h"
#include "../framework/weight.h"

#include "../framework/data_args.h"
#include "../framework/op_args.h"

#include "../../tools/time_utils.h"
#include "../utils/data_printer.h"

#include "../factory/framework_factory.h"
#include "../factory/factory.h"
#include "../factory/macro_factory.h"

using namespace cacu_tools;

namespace cacu {

class operator_base {

public:

	operator_base(blobs *&data, data_args *&args_, op_name type_) {

		s_blobs = data;
		o_blobs = NULL;
		_args = args_;
		_o_args = NULL;
		_phase = test;
		_OP_TYPE = type_;
		data->_REC();
		_weights = new vector<weight*>();
		_storage_blobs = new blobs();
		_BASE_TYPE = NULL_BASE;

	}

	operator_base(blobs *&data, op_args *&o_args_, op_name type_) {

		s_blobs = data;
		o_blobs = NULL;
		_args = NULL;
		_o_args = o_args_;
		_phase = test;
		_OP_TYPE = type_;
		data->_REC();
		_weights = new vector<weight*>();
		_storage_blobs = new blobs();
		_BASE_TYPE = NULL_BASE;

	}

	operator_base(blobs *&data, op_args *&o_args_, data_args *& args_,
			op_name type_) {

		s_blobs = data;
		o_blobs = NULL;
		_args = args_;
		_o_args = o_args_;
		_phase = test;
		_OP_TYPE = type_;
		data->_REC();
		_weights = new vector<weight*>();
		_storage_blobs = new blobs();
		_BASE_TYPE = NULL_BASE;

	}

	operator_base(blobs *&data, op_name type_) {

		s_blobs = data;
		o_blobs = NULL;
		_args = NULL;
		_o_args = NULL;
		_phase = test;
		_OP_TYPE = type_;
		data->_REC();
		_weights = new vector<weight*>();
		_storage_blobs = new blobs();
		_BASE_TYPE = NULL_BASE;

	}

	virtual ~operator_base() {

		if (_o_args != NULL) {
			delete _o_args;
			_o_args = NULL;
		}
		if (_args != NULL) {
			delete _args;
			_args = NULL;
		}
		if (_IS_ALLOC_OUTPUT) {
			if (o_blobs != NULL) {
				delete o_blobs;
				o_blobs = NULL;
			}
		}
		//clear s_blobs, but don't need to release the blob memory
		if (s_blobs != NULL) {
			delete s_blobs;
			s_blobs = NULL;
		}
		for (unsigned int i = 0; i < _weights->size(); ++i) {
			delete _weights->at(i);
			_weights->at(i) = NULL;
		}
		delete _weights;
		_weights = NULL;
		delete _storage_blobs;
		_storage_blobs = NULL;

		delete _NONE_BLOB;
	}

	virtual void check() = 0;

	virtual void init_weights() = 0;

	virtual void op(blobs *&s_blobs_, blobs *&o_blobs_) = 0;

	virtual void initial() = 0;

	virtual void grad(blobs *&s_blobs_, blobs *&o_blobs_) = 0;

	virtual void load(std::ifstream &is) = 0;

	virtual void save(std::ostream &os) = 0;

	virtual void echo() = 0;

	inline void LOOP_INIT_DATA_() {
		if (_IS_ALLOC_OUTPUT)
			o_blobs->_RESET_DATA();
		_storage_blobs->_RESET_DATA();
	}

	inline void LOOP_INIT_DIFF_() {
		if (_IS_ALLOC_OUTPUT)
			o_blobs->_RESET_DIFF();
		_storage_blobs->_RESET_DIFF();
	}

	virtual void set_phase(phase_type phase_) {
		_phase = phase_;
	}

	inline blobs *out_datas() const {
		return o_blobs;
	}

	template<typename BTYPE>
	inline BTYPE *out_data() {
		return dynamic_cast<BTYPE*>(o_blobs->at(0));
	}

	inline blobs *in_datas() const {
		return s_blobs;
	}

	template<typename BTYPE>
	inline BTYPE *in_data() const {
		return dynamic_cast<BTYPE*>(s_blobs->at(0));
	}

	inline int weights_size() const {
		return _weights->size();
	}

	inline weight *&get_weight(int i) const {
		return _weights->at(i);
	}

	inline blob *get_storage_blob(int i) const {
		if (i >= 0)
			return _storage_blobs->asblob(i);
		return _storage_blobs->asblob(_storage_blobs->size() + i);
	}

	inline vector<weight*> *&get_weights() {
		return _weights;
	}

	inline op_args *&get_op_args() {
		return _o_args;
	}

	inline data_args *&get_data_args() {
		return _args;
	}

	inline void set_is_update_weight(bool update_) {
		//LOG_INFO("Fix the operator's weight!");
		for (int i = 0; i < _weights->size(); ++i)
			_weights->at(i)->set_update(update_);
	}

	inline void set_is_variable_weight(bool variable_) {
		LOG_INFO("Set operator's weight variable == %d!", variable_);
		for (int i = 0; i < _weights->size(); ++i)
			_weights->at(i)->set_variable(variable_);
	}

	inline void infer() {
		time_utils *t = new time_utils();
//		LOG_DEBUG("Operator op: %s!",
//				macro_factory::get_op_name(_OP_TYPE).c_str());
		//forward propagation
#if __OPERATOR__TYPE__ == __STATIC_GRAPH__
		blob *blob_ = s_blobs->asblob(0);
		if (!blob_->_IS_MOTIFIED()) {
			s_blobs->_MOTIFY();
			initial();
		}
		t->start();
		//reset the data's values
		LOOP_INIT_DATA_();
		op(s_blobs, o_blobs);
//		if (macro_factory::get_op_name(_OP_TYPE) == "batch_normalize_op") {
//			cacu_print("s_data", s_blobs->asblob(0)->s_data(),
//					s_blobs->asblob(0)->length());
//			cacu_print("o_data", o_blobs->asblob(0)->s_data(),
//					o_blobs->asblob(0)->length());
//		}
#endif
		t->end();
		//LOG_DEBUG("%d operator infer time cost: %d ms", _OP_TYPE, t->get_time_span() / 1000);
		delete t;
	}

	inline void derivative() {
#if __OPERATOR__TYPE__ == __STATIC_GRAPH__
		grad(s_blobs, o_blobs);
#endif
	}

	inline void set_blob(blob_base *&blob_) {
		if (s_blobs->size() > 0) {
			s_blobs->at(0)->_CHECK_SIZE_EQ(blob_);
			switch (s_blobs->at(0)->_TYPE()) {
			case __blob__:
				delete s_blobs->asblob(0);
				break;
			case __bin_blob__:
				delete s_blobs->asbinblob(0);
				break;
			default:
				delete s_blobs->at(0);
				break;
			}
			s_blobs->at(0) = blob_;
		} else
			s_blobs->push_back(blob_);
	}

	inline void set_blobs(blobs *&blobs_) {
		if (s_blobs == NULL)
			s_blobs = blobs_;
		else {
			delete s_blobs;
			s_blobs = blobs_;
		}

	}

	inline op_name _TYPE() const {
		return _OP_TYPE;
	}

	inline bool is_alloc_output() const {
		return _IS_ALLOC_OUTPUT;
	}

	inline op_name get_optype() const {
		return _OP_TYPE;
	}

	inline base_type get_basetype() const {
		return _BASE_TYPE;
	}

	inline void register_param(weight *&w_) {
		for (int i = 0; i < _weights->size(); ++i)
			if (_weights->at(i)->name() == w_->name())
				LOG_FATAL("Invalid redefinition of parameter %s!",
						w_->name().c_str());
		_weights->push_back(w_);
	}

	void weight_pre_proc(void pfunc(weight*&), weight *&w_) {
		pfunc(w_);
	}

	inline chars_t get_op_name() {
		return _custom_op_name;
	}

	inline void set_op_name(chars_t custom_op_name_) {
		_custom_op_name = custom_op_name_;
	}

	//OpType+bool(data_args)+data_args+bool(op_args)+op_args
	void serialize(std::ostream &os) {
		int type_ = _OP_TYPE;
		os.write((char*) (&type_), sizeof(int));
		int _null_set = 0;
		if (_args != NULL) {
			_null_set = 1;
			os.write((char*) (&_null_set), sizeof(int));
			_args->serialize(os);
		} else {
			_null_set = 0;
			os.write((char*) (&_null_set), sizeof(int));
		}

		if (_o_args != NULL) {
			_null_set = 1;
			os.write((char*) (&_null_set), sizeof(int));
			_o_args->serialize(os);
		} else {
			_null_set = 0;
			os.write((char*) (&_null_set), sizeof(int));
		}
	}

	void weight_copy(operator_base *op_) {
		if (this->get_optype() != op_->get_optype())
			LOG_FATAL("Operator type mis-matched [%s vs %s]!",
					macro_factory::get_op_name(_OP_TYPE).c_str(),
					macro_factory::get_op_name(op_->get_optype()).c_str());
		if (this->get_weights()->size() != op_->get_weights()->size())
			LOG_FATAL("Operator weights are mis-matched!");
		for (unsigned int i = 0; i < this->get_weights()->size(); ++i){
			this->get_weight(i)->copy2data(op_->get_weight(i));
		}
		LOG_INFO("Finished weight copy!");
	}

protected:

	blobs *s_blobs;

	blobs *o_blobs;

	data_args *_args;

	op_args *_o_args;

	phase_type _phase;

	vector<weight*> *_weights;

	blobs *_storage_blobs;

	op_name _OP_TYPE;

	base_type _BASE_TYPE;

	chars_t _custom_op_name = "";

	const blob_body *_NONE_BLOB = new blob_body(1, 1, 1, 1);

	//create weight push_back to weights container
	inline weight* const create_param(chars_t&& name, dsize_t num,
			dsize_t channel, dsize_t width, dsize_t height, phase_type phase) {
		_weights->push_back(
				new weight(std::move(name), num, channel, width, height,
						phase));
		return _weights->back();
	}

	inline void set_param_init_type(param_init_type type, weight *&w_,
			float_t value) {
		w_->set_init_type(type, value);
	}

	inline blob_base * create_oblob(dsize_t num, dsize_t channel, dsize_t width,
			dsize_t height, phase_type phase_) {
#if __OPERATOR__TYPE__ == __DYNAMIC_GRAPH__
		return new blob(num, channel, width, height, 0, phase_, false);
#elif __OPERATOR__TYPE__ == __STATIC_GRAPH__
		return new blob(num, channel, width, height, 0, phase_);
#endif
	}

	inline blob *create_opblob(dsize_t num, dsize_t channel, dsize_t width,
			dsize_t height, phase_type phase_) {
		_storage_blobs->push_back(
				new blob(num, channel, width, height, 0, phase_));
		return dynamic_cast<blob*>(_storage_blobs->back());
	}

	inline blob *create_opblob(dsize_t num, dsize_t channel, dsize_t width,
			dsize_t height, float_t value, phase_type phase_) {
		_storage_blobs->push_back(
				new blob(num, channel, width, height, value, phase_));
		return dynamic_cast<blob*>(_storage_blobs->back());
	}

	inline bin_blob *create_bin_opblob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, phase_type phase_) {
		_storage_blobs->push_back(
				new bin_blob(num, channel, width, height, 0, phase_));
		return dynamic_cast<bin_blob*>(_storage_blobs->back());
	}

	inline bin_blob *create_bin_opblob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, unsigned int value,
			phase_type phase_) {
		_storage_blobs->push_back(
				new bin_blob(num, channel, width, height, value, phase_));
		return dynamic_cast<bin_blob*>(_storage_blobs->back());
	}

#if __USEMBEDDING__ == ON
	inline blob_base * create_em_oblob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, phase_type phase_) {
		_IS_ALLOC_OUTPUT = true;
		return new em_blob(num, channel, width, height, 0, phase_);
	}

	inline em_blob * create_em_opblob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, phase_type phase_) {
		_storage_blobs->push_back(
				new em_blob(num, channel, width, height, 0, phase_));
		return (em_blob *) _storage_blobs->back();
	}

	inline em_blob * create_em_opblob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, float_t value, phase_type phase_) {
		_storage_blobs->push_back(
				new em_blob(num, channel, width, height, value, phase_));
		return (blob *) _storage_blobs->back();
	}

	inline em_bin_blob * create_em_bin_opblob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, phase_type phase_) {
		_storage_blobs->push_back(
				new em_bin_blob(num, channel, width, height, 0, phase_));
		return (em_bin_blob *) _storage_blobs->back();
	}

	inline em_bin_blob * create_em_bin_opblob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, unsigned int value,
			phase_type phase_) {
		_storage_blobs->push_back(
				new em_bin_blob(num, channel, width, height, value, phase_));
		return (em_bin_blob *) _storage_blobs->back();
	}
#endif
	inline blobs * create_oblobs() {
		_IS_ALLOC_OUTPUT = true;
		return new blobs();
	}

	bool _NEED_BACK_PROPAGATE_FEATURE = true;

	/*
	 * load just for one time in constructed function.
	 */
	void _INIT_OP() {
		//initialize the op phase
		_phase = s_blobs->at(0)->phase();

		check();
		init_weights();
		initial();
	}

private:

	bool _IS_ALLOC_OUTPUT = false;

};

#define FACTORY_OP(CLASS) factory<CLASS>
#define MACRO_FACTORY_OP macro_factory

#define REGISTER_OP_CLASS_D(CLASS) \
	FACTORY_OP(CLASS)::register_d<CLASS> __##CLASS(CLASS_NAME(CLASS))

#define REGISTER_OP_CLASS_D_OP(CLASS) \
	FACTORY_OP(CLASS)::register_d_op<CLASS> __##CLASS(CLASS_NAME(CLASS))

#define REGISTER_OP_CLASS_D_DP(CLASS) \
	FACTORY_OP(CLASS)::register_d_dp<CLASS> __##CLASS(CLASS_NAME(CLASS))

#define REGISTER_OP_CLASS_D_ODP(CLASS) \
	FACTORY_OP(CLASS)::register_d_odp<CLASS> __##CLASS(CLASS_NAME(CLASS))

#define REGISTER_OP_MACRO(CLASS, OP_NAME) \
	MACRO_FACTORY_OP::register_macro ___##CLASS(CLASS_NAME(CLASS), OP_NAME);  \
	MACRO_FACTORY_OP::register_name ____##CLASS(CLASS_NAME(CLASS), typeid(CLASS).name())

}

#endif
