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

#include "../factory/framework_factory.h"

namespace cacu {

class operator_base {

public:

	operator_base(blob_base *&data, data_args *&args_, op_name type_) {

		s_blob = data;
		s_blobs = NULL;
		o_blob = NULL;
		o_blobs = NULL;
		_args = args_;
		_o_args = NULL;
		_phase = data->phase();
		_OP_TYPE = type_;
		data->_REC();

		_weights = new vector<weight*>();
		_storage_blobs = new blobs();
	}

	operator_base(blob_base *&data, op_name type_) {

		s_blob = data;
		s_blobs = NULL;
		o_blob = NULL;
		o_blobs = NULL;
		_args = NULL;
		_o_args = NULL;
		_phase = data->phase();
		_OP_TYPE = type_;
		data->_REC();

		_weights = new vector<weight*>();
		_storage_blobs = new blobs();
	}

	operator_base(blob_base *&data, op_args *&args_, op_name type_) {

		s_blob = data;
		s_blobs = NULL;
		o_blob = NULL;
		o_blobs = NULL;
		_args = NULL;
		_o_args = args_;
		_phase = data->phase();
		_OP_TYPE = type_;
		data->_REC();

		_weights = new vector<weight*>();
		_storage_blobs = new blobs();
	}

	operator_base(blobs *&data, data_args *&args_, op_name type_) {

		s_blob = NULL;
		s_blobs = data;
		o_blob = NULL;
		o_blobs = NULL;
		_args = args_;
		_o_args = NULL;
		_phase = data->at(0)->phase();
		_OP_TYPE = type_;
		data->_REC();

		_weights = new vector<weight*>();
		_storage_blobs = new blobs();
	}

	operator_base(blobs *&data, op_name type_) {

		s_blob = NULL;
		s_blobs = data;
		o_blob = NULL;
		o_blobs = NULL;
		_args = NULL;
		_o_args = NULL;
		_phase = data->at(0)->phase();
		_OP_TYPE = type_;
		data->_REC();

		_weights = new vector<weight*>();
		_storage_blobs = new blobs();
	}

	virtual ~operator_base() {

		delete _o_args;
		if (_IS_ALLOC_OUTPUT) {
			if (o_blob != NULL){
				delete o_blob;
				o_blob = NULL;
			}
			if (o_blobs != NULL)
				delete o_blobs;
		}
		for (unsigned int i = 0; i < _weights->size(); ++i) {
			delete _weights->at(i);
			_weights->at(i) = NULL;
		}
		delete _weights;

		delete _storage_blobs;
	}

	virtual const void check() = 0;

	virtual const void grad() = 0;

	virtual const void load(std::ifstream &is) = 0;

	virtual const void save(std::ostream &os) = 0;

	virtual const void echo() = 0;

	virtual const void LOOP_INIT_DATA_() = 0;

	virtual const void set_phase(phase_type phase_) = 0;

	inline blobs *&out_datas() {
		return o_blobs;
	}

	template<typename BTYPE>
	inline BTYPE *&out_data() const {
		return (BTYPE *&) o_blob;
	}

	inline blobs *&in_datas() {
		return s_blobs;
	}

	template<typename BTYPE>
	inline BTYPE *&in_data() const {
		return (BTYPE *&) s_blob;
	}

	inline int weights_size() const {
		return _weights->size();
	}

	inline weight* get_weight(int i) const {
		return _weights->at(i);
	}

	inline void set_is_update_weight(bool _update)
	{
		LOG_INFO("Fix the operator's weight!");
		for (int i = 0; i < _weights->size(); ++i)
			_weights->at(i)->set_update(_update);
	}

	inline void infer() {

		blob_base *blob_ = (s_blobs == NULL) ? s_blob : s_blobs->at(0);
		if (!blob_->_IS_MOTIFIED()) {
			(s_blobs == NULL) ? blob_->_MOTIFY() : s_blobs->_MOTIFY();
			initial();
		}
		//reset the data's values
		LOOP_INIT_DATA_();
		//forward propagation
		op();
	}

	inline void set_blob(blob_base *&blob_) {
		s_blob->_CHECK_SIZE_EQ(blob_);
		s_blob = blob_;
	}

	inline void set_blobs(blobs *&blobs_) {
		s_blobs = blobs_;
	}

	inline op_name _TYPE() {
		return _OP_TYPE;
	}

protected:

	blobs *s_blobs;

	blob_base *s_blob;

	blob_base *o_blob;

	blobs *o_blobs;

	data_args *_args;

	op_args *_o_args;

	phase_type _phase;

	vector<weight*> *_weights;

	blobs *_storage_blobs;

	op_name _OP_TYPE;

	//create weight push_back to weights container
	inline weight* const create_param(chars_t name, dsize_t num,
			dsize_t channel, dsize_t width, dsize_t height, phase_type phase) {
		_add2op_weights(new weight(name, num, channel, width, height, phase));
		return _weights->back();
	}

	inline void set_param_init_type(param_init_type type, weight *w_,
			float_t value) {
		w_->set_init_type(type, value);
	}

	inline void _add2op_weights(weight *w) {
		_weights->push_back(w);
	}

	inline blob_base * create_oblob(dsize_t num, dsize_t channel, dsize_t width,
			dsize_t height, phase_type phase_) {
		_IS_ALLOC_OUTPUT = true;
		return new blob(num, channel, width, height, 0, phase_);
	}

	inline blob *create_opblob(dsize_t num, dsize_t channel, dsize_t width,
			dsize_t height, phase_type phase_) {
		_storage_blobs->push_back(
				new blob(num, channel, width, height, 0, phase_));
		return (blob *) _storage_blobs->back();
	}

	inline blob *create_opblob(dsize_t num, dsize_t channel, dsize_t width,
			dsize_t height, float_t value, phase_type phase_) {
		_storage_blobs->push_back(
				new blob(num, channel, width, height, value, phase_));
		return (blob *) _storage_blobs->back();
	}

	inline bin_blob *create_bin_opblob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, phase_type phase_) {
		_storage_blobs->push_back(
				new bin_blob(num, channel, width, height, 0, phase_));
		return (bin_blob *) _storage_blobs->back();
	}

	inline bin_blob *create_bin_opblob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, unsigned int value,
			phase_type phase_) {
		_storage_blobs->push_back(
				new bin_blob(num, channel, width, height, value, phase_));
		return (bin_blob *) _storage_blobs->back();
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

	virtual const void init_weights() = 0;

	virtual const void op() = 0;

	virtual const void initial() = 0;

private:

	bool _IS_ALLOC_OUTPUT = false;

};

}


#endif