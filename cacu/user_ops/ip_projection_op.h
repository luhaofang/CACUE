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

#ifndef IP_PROJECTION_OP_H
#define IP_PROJECTION_OP_H


#include "../ops/ops_definition.h"
#include "../ops/operator_base.h"

namespace cacu {

class ip_projection_op: public operator_base {

public:

	ip_projection_op(blobs *&data, op_args *&o_args_, data_args *&args_);

	~ip_projection_op();

	void initial() override;

	void init_weights() override;

	void check() override;

	void op(blobs *&s_blobs_,blobs *&o_blobs_) override;

	void grad(blobs *&s_blobs_,blobs *&o_blobs_) override;

	void load(std::ifstream& is) override;

	void save(std::ostream& os) override;

	void echo() override;

	inline void set_weight_init_type(param_init_type _type,
			float_t value = 0.0) {
		for(int i = 0 ; i < _ws->size(); ++i)
			set_param_init_type(_type, _ws->at(i), value);
	}

	inline void set_bias_init_type(param_init_type _type, float_t value = 0.0) {
		for(int i = 0 ; i < _ws->size(); ++i)
			set_param_init_type(_type, _biases->at(i), value);
	}

	void set_is_use_bias(bool switcher_) {
		_is_use_bias = switcher_;
	}

	//set spectral norm params
	inline void set_is_norm_weights(bool is_norm_weights_, int pow_iter_ = 1){
		_IS_NORM_WEIGHTS = is_norm_weights_;
		//if normalize weight, the operator should not use bias
		if (_IS_NORM_WEIGHTS == true)
			_is_use_bias = false;
		CHECK_GE_OP(pow_iter_, 1, "pow iteration must > 1 vs %d!", pow_iter_);
		_pow_iter = pow_iter_;
	}


private:

	//inner_product_op use bias switcher
	bool _is_use_bias = true;

	vector<weight*> *_ws = NULL;

	vector<weight*> *_biases = NULL;

	blob *_bias_multiplier = NULL;

	bool _IS_NORM_WEIGHTS = false;

	blobs* _us = NULL;
	blobs* _vs = NULL;
	blobs* _ws_original = NULL;

	int _pow_iter = 1;

	void norm_weight();

	void norm_recover();

};
}


#endif
