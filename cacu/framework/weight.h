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

#ifndef WEIGHT_H_
#define WEIGHT_H_

#include "blob.h"

#include "../tensor/cuda/cuda_utils.h"

namespace cacu {

class weight: public blob {

public:

	weight(chars_t name, dsize_t num, dsize_t channel, dsize_t width,
			dsize_t height, phase_type phase);

	~weight();

	inline chars_t name() const {
		return _name;
	}

	inline void set_lr(float_t lr_) {
		_update_lr = lr_;
	}

	inline float_t lr() const {
		return _update_lr;
	}

	inline void set_decay(float_t decay_mult_) {
		_decay_mult = decay_mult_;
	}

	inline float_t decay() const {
		return _decay_mult;
	}

	inline void set_update(bool update_) {
		_update = update_;
	}

	inline bool update() const {
		return _update;
	}

	void set_init_type(param_init_type type, float_t value);

	/*
	 * serializa blob data, output data to model file
	 */
	void serializa_group(std::ostream& os, int group);

	/*
	 * loads blob data from model file
	 */
	void load_group(std::ifstream& is, int group);

	/*
	 * loads blob data from model file
	 */
	void load(std::ifstream& is) override;

private:

	chars_t _name;

	float_t _update_lr;

	float_t _decay_mult;

	bool _update;

};

}


#endif
