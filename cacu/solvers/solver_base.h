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

#ifndef SOLVER_BASE_H_
#define SOLVER_BASE_H_

#include "../math/math_definition.h"
#include "../framework/weight.h"
#include "../framework/network.h"


namespace cacu {

class solver_base {

public:

	solver_base(network *&net_);

	virtual ~solver_base();

	inline void set_weight_decay(float_t weight_decay_) {
		_global_weight_decay = weight_decay_;
	}

	inline void set_lr(float_t lr_) {
		_global_lr = lr_;
	}

	inline void set_regularize(regularize_type type_) {
		_regularize = type_;
	}

	inline float_t weight_decay() const {
		return _global_weight_decay;
	}

	inline float_t lr() const {
		return _global_lr;
	}

	inline regularize_type regularize() const {
		return _regularize;
	}

	/*
	 * where weight_index denote the weight's id in sovler's vector
	 */
	virtual void update_weight(weight* w_, int weight_index_) = 0;

	void crop_grad(blob* g_);

	void train_iter();

	/**
	 * change global_lr by rate after several training iterations
	 *
	 */
	inline void set_lr_iter(float_t lr_rate_) {
		this->_global_lr *= lr_rate_;
	}

protected:

	float_t _global_lr = 1.0;

	float_t _global_weight_decay = 0.004;

	regularize_type _regularize = L2;

	network *_net;

	/**
	 * add regular to gradient
	 * where i is the index of _w
	 */
	void __REGULARIZE__(weight *w_, int weight_index_);
	/**
	 * normalize gradient
	 * where i is the index of _w
	 */
	void __NORMALIZE__(weight *w_);

private:

	blobs* _temp;

	int _batch_size;

};
}


#endif