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

#include "relu_op.h"
#include "../math/math.h"

namespace cacu {

relu_op::relu_op(blobs *&data) :
		activate_base_op(data, CACU_RELU) {
	_INIT_OP();
}

relu_op::~relu_op() {

}

void relu_op::initial() {
	if (o_blobs == NULL)
		o_blobs = s_blobs;
	else
		o_blobs->_NEED_MOTIFY();
//#if __USE_CUDNN__ == ON
//	blob *s_blob_ = s_blobs->asblob(0);
//	if (s_blob_->is_malloced()) {
//		set_activation_desc(_activation_desc, CUDNN_ACTIVATION_RELU,
//				CUDNN_PROPAGATE_NAN, float_t(0));
//	}
//#endif

}

void relu_op::op(blobs *&s_blobs_, blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
	em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
	em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

	cacu_relu_cpu(o_blob_->s_data(), o_blob_->count());

#else

	o_blobs_ = s_blobs_;
	blob *s_blob_ = s_blobs_->asblob(0);

//#if __USE_CUDNN__ == OFF
//	cacu_relu(s_blob_->s_data(), s_blob_->count());
//
//#else
//	cudnn_activation_forward(_activation_handle, _activation_desc,
//			s_blob_->tensor_desc(), s_blob_->s_data(), s_blob_->tensor_desc(),
//			s_blob_->s_data());
//#endif
	cacu_relu(s_blob_->s_data(), s_blob_->count());

#endif
}

void relu_op::grad(blobs *&s_blobs_, blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
	em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
	em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

	cacu_relu_grad_cpu(s_blob_->s_data(), o_blob_->s_diff(),
			s_blob_->count());

#else
	o_blobs_ = s_blobs_;
	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);

//#if __USE_CUDNN__ == OFF
//	cacu_relu_grad(s_blob_->s_data(), o_blob_->s_diff(), s_blob_->count());
//#else
//	cudnn_activation_backward(_activation_handle, _activation_desc,
//			s_blob_->tensor_desc(), s_blob_->s_data(), s_blob_->tensor_desc(),
//			s_blob_->s_diff(), o_blob_->tensor_desc(), o_blob_->s_data(),
//			o_blob_->tensor_desc(), o_blob_->s_diff());
//#endif
	cacu_relu_grad(s_blob_->s_data(), o_blob_->s_diff(), s_blob_->count());

#endif
}

void relu_op::echo() {
	LOG_INFO("relu op: in(%d, %d, %d), out(%d, %d, %d)",
			s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
			s_blobs->at(0)->height(), o_blobs->at(0)->channel(),
			o_blobs->at(0)->width(), o_blobs->at(0)->height());
}

REGISTER_OP_CLASS_D(relu_op);
REGISTER_OP_MACRO(relu_op, CACU_RELU);

}
