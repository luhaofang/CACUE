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

#ifndef LOSS_BASE_OP_H_
#define LOSS_BASE_OP_H_

#include "../math/math_definition.h"

namespace cacu {

class loss_base_op: public operator_base {

public:

	loss_base_op(blobs *&data, op_name type_) :
		operator_base(data, type_) {
		_BASE_TYPE = LOSS_BASE;
	}

	loss_base_op(blobs *&data, op_args *&args_, op_name type_) :
		operator_base(data, args_, type_) {
		_BASE_TYPE = LOSS_BASE;
	}

	~loss_base_op() {

	}

	void init_weights() final override {
		return;
	}

	void load(std::ifstream& is) final override {
		return;
	}

	void save(std::ostream& os) final override {
		return;
	}

	void echo() final override
	{
		LOG_INFO("loss : %f", _loss / _loss_weight);
		if (_loss_weight != 1.0)
			LOG_INFO("weighted loss : %f", _loss);
	}

	float_t normalizer() const {
		blob_base* blob_ = s_blobs->at(0);
		return (_loss_weight / (float_t)blob_->num());
	}

	inline float_t loss() {
		return _loss;
	}

	inline void set_loss_weight(float_t weight_) {
		CHECK_GT_OP(weight_, 0, "loss weight must > 0 vs %f", weight_);
		_loss_weight = weight_;
	}

	inline float_t loss_weight() const {
		return _loss_weight;
	}
//
//	inline void set_direction(grad_direction_type direction_){
//		_direction = direction_;
//		switch(_direction)
//		{
//			case minimize:
//				_loss_direction = 1.0;
//				break;
//			case maximize:
//				_loss_direction = -1.0;
//				break;
//			default:
//				_loss_direction = 1.0;
//				break;
//		}
//	}

protected:

	float_t _loss = 0.0;

	float_t _loss_weight = 1.0;
};
}


#endif
