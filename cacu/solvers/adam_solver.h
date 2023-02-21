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


#ifndef ADAM_SOLVER_H_
#define ADAM_SOLVER_H_

#include "solver_base.h"

namespace cacu {

class adam_solver: public solver_base {

public:

	adam_solver(network *&net_);

	~adam_solver();

	inline void set_epsilon(float_t epsilon_) {
		_epsilon = epsilon_;
	}

	inline float_t epsilon() const {
		return _epsilon;
	}

	inline void set_beta2(float_t beta_) {
		_beta2 = beta_;
	}

	inline float_t beta2() const {
		return _beta2;
	}

	inline void set_beta1(float_t beta_) {
		_beta1 = beta_;
	}

	inline float_t beta1() const {
		return _beta1;
	}

	inline void echo() {

	}

	void load_param(const chars_t& config_);

protected:

	/**
	 * update weight value
	 * where weight_index_ is the weight index in _history_v
	 */
	void update_weight(weight *&w_, int weight_index_, int step_);

private:


	blobs* _history_m;

	blobs* _history_v;

	float_t _epsilon = 1E-8;

	float_t _beta1 = 0.9;

	float_t _beta2 = 0.999;

};
}



#endif /* ADAM_SOLVER_H_ */
