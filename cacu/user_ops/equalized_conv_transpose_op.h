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

#ifndef equalized_conv_transpose_op_H_
#define equalized_conv_transpose_op_H_

#include "../../tools/serializer_utils.h"

#include "../ops/ops_definition.h"
#include "../ops/operator_base.h"
#include "../ops/conv_base_op.h"

#include <math.h>

using namespace cacu_tools;

namespace cacu {

class equalized_conv_transpose_op: public conv_base_op {

public:

	//output_channel, kernel_size, stride, pad, input_dim, channel
	//if input_size == 1, use arguement pad to calculate the output size
	equalized_conv_transpose_op(blobs *&data, data_args *&args_);

	~equalized_conv_transpose_op();

	void initial() override;

	void init_weights() override;

	void op(blobs *&s_blobs_,blobs *&o_blobs_) override;

	void grad(blobs *&s_blobs_,blobs *&o_blobs_) override;

	void echo() override;

	void set_lr_mul(float_t lr_mul_)
	{
		_lr_mul = lr_mul_;
	}

	void set_use_wscale(bool use_wscale_)
	{
		_use_wscale = use_wscale_;
		if(_use_wscale){
			_std_init = 1.0 / _lr_mul;
			_w_mul = sqrtf((float_t)(2. / (_args->output_channel()
					*_args->kernel_size()*_args->kernel_size()))) * _lr_mul;
		}
		else{
			_std_init = sqrtf((float_t)(2. / (_args->output_channel()
					*_args->kernel_size()*_args->kernel_size()))) / _lr_mul;
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

	int _pad_w = 0;
	int _pad_h = 0;

};
}


#endif
