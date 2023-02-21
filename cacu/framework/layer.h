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

#ifndef LAYER_H_
#define LAYER_H_

#include "../ops/ops_definition.h"

#include "layer_base.h"

namespace cacu {

class layer: public layer_base {

public:

	layer(data_args *args);

	layer();

	~layer();

	int caculate_data_space() {
		return 0;
	}

	layer* op(op_name op_);

	layer* op(op_name op_, blob_base *blob_);

	layer* op(op_name op_, blob_base *blob_, data_args * args_);

	layer* op(op_name op_, blob_base *blob_, op_args *args_);

	layer* op(op_name op_, blob_base *blob_, op_args *o_args_, data_args *args_);

	layer* op(op_name op_, blobs *blobs_);

	layer* op(op_name op_, blobs *blobs_, data_args *args_);

	layer* op(op_name op_, op_args *args_);

	layer* op(op_name op_, blobs *blobs_, op_args *o_args_);

	layer* op(op_name op_, blobs *blobs_, op_args *o_args_, data_args *args_);

	template<typename OPTYPE>
	layer* op() {
		blobs *blobs_ = cacu_allocator::create_blobs();
		if (out_blob != NULL)
			blobs_->push_back(out_blob);
		add_op(operator_factory::create_op<OPTYPE>(
				MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()), blobs_, _args, NULL));
		out_blob = _ops->back()->out_data<blob_base>();
		_ops->back()->echo();
		return this;
	}

	template<typename OPTYPE>
	layer* op(blob_base *blob_) {
		if (blob_ == NULL)
			LOG_FATAL("input data is NULL!");
		blobs *blobs_ = cacu_allocator::create_blobs();
		if (out_blob != NULL)
			blobs_->push_back(out_blob);
		blobs_->push_back(blob_);
		add_op(operator_factory::create_op<OPTYPE>(
				MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()), blobs_, _args, NULL));
		out_blob = _ops->back()->out_data<blob_base>();
		_ops->back()->echo();
		return this;
	}

	template<typename OPTYPE>
	layer* op(blob_base *blob_, data_args *args_) {
		if (_args != NULL && args_ != NULL)
			LOG_FATAL("layer data arguments is already defined, if your want a new data arguments, please setup a new layer!");
		if (blob_ == NULL)
			LOG_FATAL("input data is NULL!");
		if (_args == NULL)
			_args = args_;
		blobs *blobs_ = cacu_allocator::create_blobs();
		if (out_blob != NULL)
			blobs_->push_back(out_blob);
		blobs_->push_back(blob_);
		add_op(operator_factory::create_op<OPTYPE>(
				MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()), blobs_, _args, NULL));
		out_blob = _ops->back()->out_data<blob_base>();
		_ops->back()->echo();
		return this;
	}

	template<typename OPTYPE>
	layer* op(blob_base * blob_, op_args *o_args_, data_args *args_) {
		if (_args != NULL && args_ != NULL)
			LOG_FATAL("layer data arguments is already defined, if your want a new data arguments, please setup a new layer!");
		if (blob_ == NULL)
			LOG_FATAL("input data is NULL!");
		if (_args == NULL)
			_args = args_;
		blobs *blobs_ = cacu_allocator::create_blobs();
		if (out_blob != NULL)
			blobs_->push_back(out_blob);
		blobs_->push_back(blob_);
		add_op(operator_factory::create_op<OPTYPE>(
				MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()), blobs_, _args, o_args_));
		out_blob = _ops->back()->out_data<blob_base>();
		_ops->back()->echo();
		return this;
	}

	template<typename OPTYPE>
	layer* op(blobs *blobs_) {

		if (blobs_ == NULL)
			LOG_FATAL("input data is NULL!");
		if (out_blob != NULL)
		{
			vector<blob_base*> blob_data;
			for(unsigned int i = 0 ; i < blobs_->size(); ++i)
				blob_data.push_back(blobs_->at(i));
			blobs_->clear();
			blobs_->push_back(out_blob);
			for(unsigned int i = 0 ; i < blob_data.size(); ++i)
				blobs_->push_back(blob_data[i]);
		}
		add_op(operator_factory::create_op<OPTYPE>(
				MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()), blobs_, _args, NULL));
		out_blob = _ops->back()->out_data<blob_base>();
		_ops->back()->echo();
		return this;
	}

	template<typename OPTYPE>
	layer* op(blobs *blobs_, data_args *args_) {
		if (_args != NULL && args_ != NULL)
			LOG_FATAL("layer data arguments is already defined, if your want a new data arguments, please setup a new layer!");
		if (blobs_ == NULL)
			LOG_FATAL("input data is NULL!");
		if (_args == NULL)
			_args = args_;
		if (out_blob != NULL)
		{
			vector<blob_base*> blob_data;
			for(unsigned int i = 0 ; i < blobs_->size(); ++i)
				blob_data.push_back(blobs_->at(i));
			blobs_->clear();
			blobs_->push_back(out_blob);
			for(unsigned int i = 0 ; i < blob_data.size(); ++i)
				blobs_->push_back(blob_data[i]);
		}
		add_op(operator_factory::create_op<OPTYPE>(
				MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()), blobs_, _args, NULL));
		out_blob = _ops->back()->out_data<blob_base>();
		_ops->back()->echo();
		return this;
	}

	template<typename OPTYPE>
	layer* op(blob_base *blob_, op_args *args_) {
		if (blob_ == NULL)
			LOG_FATAL("input data is NULL!");
		blobs *blobs_ = cacu_allocator::create_blobs();
		if (out_blob != NULL)
			blobs_->push_back(out_blob);
		blobs_->push_back(blob_);
		add_op(operator_factory::create_op<OPTYPE>(
				MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()), blobs_, _args, args_));
		out_blob = _ops->back()->out_data<blob_base>();
		_ops->back()->echo();
		return this;
	}

	template<typename OPTYPE>
	layer* op(op_args *args_) {
		blobs *blobs_ = cacu_allocator::create_blobs();
		if (out_blob != NULL)
			blobs_->push_back(out_blob);
		add_op(operator_factory::create_op<OPTYPE>(
				MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()), blobs_, _args, args_));
		out_blob = _ops->back()->out_data<blob_base>();
		_ops->back()->echo();
		return this;
	}

	template<typename OPTYPE>
	layer* op(blobs *blobs_, op_args *args_) {
		if (blobs_ == NULL)
			LOG_FATAL("input data is NULL!");

		if (out_blob != NULL)
		{
			vector<blob_base*> blob_data;
			for(unsigned int i = 0 ; i < blobs_->size(); ++i)
				blob_data.push_back(blobs_->at(i));
			blobs_->clear();
			blobs_->push_back(out_blob);
			for(unsigned int i = 0 ; i < blob_data.size(); ++i)
				blobs_->push_back(blob_data[i]);
		}
		add_op(operator_factory::create_op<OPTYPE>(
				MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()), blobs_, _args, args_));
		out_blob = _ops->back()->out_data<blob_base>();
		_ops->back()->echo();
		return this;
	}

	template<typename OPTYPE>
	layer* op(blobs * blobs_, op_args *o_args_, data_args *args_) {
		if (_args != NULL && args_ != NULL)
			LOG_FATAL("layer data arguments is already defined, if your want a new data arguments, please setup a new layer!");
		if (blobs_ == NULL)
			LOG_FATAL("input data is NULL!");
		if (_args == NULL)
			_args = args_;
		if (out_blob != NULL)
		{
			vector<blob_base*> blob_data;
			for(unsigned int i = 0 ; i < blobs_->size(); ++i)
				blob_data.push_back(blobs_->at(i));
			blobs_->clear();
			blobs_->push_back(out_blob);
			for(unsigned int i = 0 ; i < blob_data.size(); ++i)
				blobs_->push_back(blob_data[i]);
		}
		add_op(operator_factory::create_op<OPTYPE>(
				MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()), blobs_, args_, o_args_));
		out_blob = _ops->back()->out_data<blob_base>();
		_ops->back()->echo();
		return this;
	}

	/*
	 * This function will be removed in the next version.
	 * Duplicated method!
	 */
	template<typename OPTYPE>
	inline OPTYPE *get_op(int i, op_name optype_) const {
		if(i >= _ops->size())
			LOG_FATAL("Index id must less than ops size!");
		if(optype_ == _ops->at(i)->_TYPE()){
			OPTYPE *op = dynamic_cast<OPTYPE*>(_ops->at(i));
			if(op == NULL)
				LOG_FATAL("op type casting failed!");
			return op;
		}
		else{
			LOG_FATAL("Use a wrong type operator casting as %s!",
					MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()).c_str());
		}
	}

	template<typename OPTYPE>
	inline OPTYPE *get_op(int i) const {
		if(i >= _ops->size())
			LOG_FATAL("Index id must less than ops size!");
		if(MACRO_FACTORY_OP::get_op_type(MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name())) == _ops->at(i)->_TYPE()){
			OPTYPE *op = dynamic_cast<OPTYPE*>(_ops->at(i));
			if(op == NULL)
				LOG_FATAL("op type casting failed!");
			return op;
		}
		else{
			LOG_FATAL("Use a wrong type operator casting as %s!",
					MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()).c_str());
		}
	}

	inline operator_base *get_head_op() const{
		return _ops->at(0);
	}

	inline operator_base *get_out_op() const{
		assert(_ops->size() > 0);
		return _ops->at(_ops->size() - 1);
	}

	inline blob_base *get_oblob() {
		return out_blob;
	}

	inline blob *oblob()
	{
		return dynamic_cast<blob*>(get_oblob());
	}

	inline blobs *get_oblobs() const {
		return _ops->back()->out_datas();
	}

private:


};

}


#endif
