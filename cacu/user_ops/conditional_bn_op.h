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

#ifndef CONDITIONAL_BN_OP_H_
#define CONDITIONAL_BN_OP_H_

//#include "../ops/ops_definition.h"
#include "../ops/bn_base_op.h"

namespace cacu {

class conditional_bn_op: public bn_base_op {

public:

	conditional_bn_op(blobs *&data, op_args *&args_, data_args *&d_args_);

	~conditional_bn_op();

	void initial() override;

	void init_weights() override;

	void check() override;

	void op(blobs *&s_blobs_, blobs *&o_blobs_) override;

	void grad(blobs *&s_blobs_, blobs *&o_blobs_) override;

	void load(std::ifstream& is) override;

	void save(std::ostream& os) override;

	void echo() override;

	inline void set_scale_init_type(param_init_type _type,
			float_t value = 0.0) {
		for (int i = 0; i < _scales->size(); ++i)
			set_param_init_type(_type, _scales->at(i), value);
	}

	void set_scale_init_type(param_init_type _type, float_t mean, float_t std);

	inline void set_shift_init_type(param_init_type _type,
			float_t value = 0.0) {
		for (int i = 0; i < _shifts->size(); ++i)
			set_param_init_type(_type, _shifts->at(i), value);
	}

	void update_history() override;

	float_t _moving_average_fraction = 0.9;

	float_t epsilon = 0.00001;

	void set_phase(phase_type phase_) override;

	inline void set_is_use_global_stats(bool use_global_stats_) {
		//LOG_INFO("Use global stats: %d",use_global_stats_);
		use_global_stats = use_global_stats_;
	}

	inline void set_interpolated_shifting_rate(
			float_t interpolated_shifting_rate_) {
		_interpolated_shifting_rate = interpolated_shifting_rate_;
	}

	inline vector<weight*>* get_scales() const {
		return _scales;
	}

	inline vector<weight*>* get_shifts() const {
		return _shifts;
	}

	/*
	 * set morphing step and compute parameters
	 */
	void morphing_step(int step, int cls1, int cls2);

private:

	vector<weight*> *_scales = NULL;

	vector<weight*> *_shifts = NULL;

	blob *_mean = NULL;

	blob *_var = NULL;

	blob *_std = NULL;

	blob *_history_mean = NULL;

	blob *_history_var = NULL;

	blob *_dim_sum = NULL;

	blob *_x = NULL;

	blob *_moving_scalar = NULL;

	blob *_one = NULL;
	blob *_mutipler = NULL;
	blob *_num_mutipler = NULL;

	vec_i label_data_;

	bool use_global_stats = false;

	float_t _interpolated_shifting_rate = 0;

	int _morphing_step = 0;

	blob *_morphing_scale = NULL;

	blob *_morphing_shift = NULL;

};
}

#endif
