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

#ifndef equalized_inner_product_op_H_
#define equalized_inner_product_op_H_

#include "../ops/ops_definition.h"
#include "../ops/operator_base.h"

#include <math.h>

namespace cacu {

class equalized_inner_product_op: public operator_base {

public:

	equalized_inner_product_op(blobs *&data, data_args *&args_);

	~equalized_inner_product_op();

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
		set_param_init_type(_type, _w, value);
	}

	inline void set_bias_init_type(param_init_type _type, float_t value = 0.0) {
		set_param_init_type(_type, _bias, value);
	}

	void set_is_use_bias(bool switcher_) {
		_is_use_bias = switcher_;
	}

	void set_lr_mul(float_t lr_mul_)
	{
		_lr_mul = lr_mul_;
	}

	void set_use_wscale(bool use_wscale_)
	{
		_use_wscale = use_wscale_;
		if(_use_wscale){
			_std_init = 1.0 / _lr_mul;
			_w_mul = sqrtf((float_t)(2. / _args->channel())) * _lr_mul;
		}
		else{
			_std_init = sqrtf((float_t)(2. / _args->channel())) / _lr_mul;
			_w_mul = _lr_mul;
		}
		_b_mul = _lr_mul;
//		LOG_DEBUG("%f, %f, %f", _std_init, _w_mul, _b_mul);
	}

	inline void std_init_w() const {
		_w->set_init_type(gaussian, _std_init);
	}

private:

	float_t _lr_mul = 1.;

	float_t _w_mul = 1.;

	float_t _b_mul = 1.;

	float_t _std_init = 1.;

	bool _use_wscale = false;

	//equalized_inner_product_op use bias switcher
	bool _is_use_bias = true;

	weight *_w = NULL;

	weight *_bias = NULL;

	blob *_bias_multiplier = NULL;

};
}


#endif
