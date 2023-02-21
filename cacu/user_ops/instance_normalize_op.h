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

#ifndef INSTANCE_NORMALIZE_OP_HPP_
#define INSTANCE_NORMALIZE_OP_HPP_


#include "../ops/ops_definition.h"
#include "../ops/operator_base.h"
#include "../ops/bn_base_op.h"


namespace cacu {

class instance_normalize_op: public bn_base_op {

public:

	instance_normalize_op(blobs *&data, data_args *&args_);

	~instance_normalize_op();

	void initial() override;

	void init_weights() override;

	void check() override;

	void op(blobs *&s_blobs_,blobs *&o_blobs_) override;

	void grad(blobs *&s_blobs_,blobs *&o_blobs_) override;

	void load(std::ifstream& is) override;

	void save(std::ostream& os) override;

	void echo() override;

	void update_history() override;

	inline weight* scale() {
		return _scale;
	}

	inline weight* shift() {
		return _shift;
	}

	inline void set_scale_init_type(param_init_type _type,
			float_t value = 0.0) {
		set_param_init_type(_type, _scale, value);
	}

	void set_scale_init_type(param_init_type _type, float_t mean,
				float_t std);

	inline void set_shift_init_type(param_init_type _type,
			float_t value = 0.0) {
		set_param_init_type(_type, _shift, value);
	}

	float_t _epsilon = 1e-8;

	void set_phase(phase_type phase_) override;

private:

	weight *_scale = NULL;

	weight *_shift = NULL;

	blob *_mean = NULL;

	blob *_var = NULL;

	blob *_std = NULL;

	blob *_x = NULL;

	blob *_moving_scalar = NULL;

	blob *_one = NULL;
	blob *_mutipler = NULL;

#if __USE_CUDNN__ == ON
	cudnnBatchNormMode_t _bn_mode = CUDNN_BATCHNORM_SPATIAL;

	cudnnHandle_t _in_handle;

	cudnnTensorDescriptor_t _data_desc = NULL;

	cudnnTensorDescriptor_t _mean_desc = NULL;

#endif

};
}


#endif
