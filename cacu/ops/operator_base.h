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

#include "../factory/framework_factory.h"
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
#if __OPERATOR__TYPE__ == __STATIC_GRAPH__
		data->_REC();
#endif
		_weights = new vector<weight*>();
		_storage_blobs = new blobs();

	}

	operator_base(blobs *&data, op_args *&o_args_, op_name type_) {

		s_blobs = data;
		o_blobs = NULL;
		_args = NULL;
		_o_args = o_args_;
		_phase = test;
		_OP_TYPE = type_;
#if __OPERATOR__TYPE__ == __STATIC_GRAPH__
		data->_REC();
#endif
		_weights = new vector<weight*>();
		_storage_blobs = new blobs();

	}

	operator_base(blobs *&data, op_name type_) {

		s_blobs = data;
		o_blobs = NULL;
		_args = NULL;
		_o_args = NULL;
		_phase = test;
		_OP_TYPE = type_;
#if __OPERATOR__TYPE__ == __STATIC_GRAPH__
		data->_REC();
#endif
		_weights = new vector<weight*>();
		_storage_blobs = new blobs();

	}

	virtual ~operator_base() {

		delete _o_args;
		if (_IS_ALLOC_OUTPUT) {
			if (o_blobs != NULL)
			{
				delete o_blobs;
				o_blobs = NULL;
			}
		}
		//clear s_blobs, but don't need to release the blob memory
		if( s_blobs != NULL){
			s_blobs->clear();
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
	}

	virtual void check() = 0;

	virtual void init_weights() = 0;

	virtual void op(blobs *s_blobs_, blobs *o_blobs_) = 0;

	virtual void initial() = 0;

	virtual void grad(blobs *s_blobs_, blobs *o_blobs_) = 0;

	virtual void load(std::ifstream &is) = 0;

	virtual void save(std::ostream &os) = 0;

	virtual void echo() = 0;

	inline void LOOP_INIT_DATA_(){
		if(_IS_ALLOC_OUTPUT)
			o_blobs->_RESET_DATA();
		_storage_blobs->_RESET_DATA();
		for(int i = 0; i < _weights->size(); ++i)
			_weights->at(i)->_RESET_DIFF();
	}

	virtual void set_phase(phase_type phase_) = 0;

	inline blobs *&out_datas() const{
		return (blobs *&)o_blobs;
	}

	template<typename BTYPE>
	inline BTYPE *&out_data() const {
		return (BTYPE *&) o_blobs->at(0);
	}

	inline blobs *&in_datas() const{
		return (blobs *&)s_blobs;
	}

	template<typename BTYPE>
	inline BTYPE *&in_data() const {
		return (BTYPE *&) s_blobs->at(0);
	}

	inline int weights_size() const {
		return _weights->size();
	}

	inline weight *& get_weight(int i) const {
		return _weights->at(i);
	}

	inline void set_is_update_weight(bool _update)
	{
		//LOG_INFO("Fix the operator's weight!");
		for (int i = 0; i < _weights->size(); ++i)
			_weights->at(i)->set_update(_update);
	}

	inline void infer() {
		time_utils *t = new time_utils();
		//forward propagation
#if __OPERATOR__TYPE__ == __STATIC_GRAPH__
		blob_base *blob_ = s_blobs->at(0);
		if (!blob_->_IS_MOTIFIED()) {
			s_blobs->_MOTIFY();
			initial();
		}
		t->start();
		//reset the data's values
		LOOP_INIT_DATA_();
		op(s_blobs, o_blobs);
#endif
		t->end();
		//LOG_DEBUG("%d operator infer time cost: %d ms", _OP_TYPE, t->get_time_span() / 1000);
		delete t;
	}

	inline void derivative()
	{
#if __OPERATOR__TYPE__ == __STATIC_GRAPH__
		grad(s_blobs, o_blobs);
#endif
	}

	inline void set_blob(blob_base *&blob_) {
		if(s_blobs->size() > 0){
			s_blobs->at(0)->_CHECK_SIZE_EQ(blob_);
			switch(s_blobs->at(0)->_TYPE()){
			case __blob__:
				delete (blob*)s_blobs->at(0);
				break;
			case __bin_blob__:
				delete (bin_blob*)s_blobs->at(0);
				break;
			default:
				delete s_blobs->at(0);
				break;
			}
			s_blobs->at(0) = blob_;
		}
		else
			s_blobs->push_back(blob_);
	}

	inline void set_blobs(blobs *&blobs_) {
		if(s_blobs == NULL)
			s_blobs = blobs_;
		else
		{
			delete s_blobs;
			s_blobs = blobs_;
		}

	}

	inline op_name _TYPE() {
		return _OP_TYPE;
	}


	/*
	 * if using dynamic graph computing, this function is used to alloc the operator.
	 */
	void alloc_create_op(){

#if __OPERATOR__TYPE__ == __DYNAMIC_GRAPH__
		check();
		initial();
		if(_weights->size() == 0)
			init_weights();
#endif
	}

	/*
	 * if the input s_blobs is not assigned when it's decleared.
	 * init_sblob used to register s_blob data.
	 */
	inline void init_sblob(){
		if(s_blobs == NULL)
			LOG_FATAL("Operator input data is NULL!");
		_phase = s_blobs->at(0)->phase();
	}

	//inline void __NEED_BACK_PROPAGATE__(bool need_back_propagate_){
	//	_NEED_BACK_PROPAGATE_FEATURE = need_back_propagate_;
	//}

protected:

	blobs *s_blobs;

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
		_weights->push_back(new weight(name, num, channel, width, height, phase));
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

	bool _NEED_BACK_PROPAGATE_FEATURE = true;

	/*
	 * load just for one time in constructed function.
	 */
	void _INIT_OP(){
#if __OPERATOR__TYPE__ == __STATIC_GRAPH__
		//initialize the op phase
		_phase = s_blobs->at(0)->phase();
		check();
		initial();
		init_weights();
#endif
	}


private:

	bool _IS_ALLOC_OUTPUT = false;

};

}


#endif
