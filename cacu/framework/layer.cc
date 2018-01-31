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

#include "layer.h"

namespace cacu {

layer::layer(size_t output_channel, size_t kernel_size, size_t stride,
		size_t pad, size_t input_dim, size_t channel) :
		layer_base(output_channel, kernel_size, stride, pad, input_dim, channel) {

}

layer* layer::op(op_name op_) {

	blobs *blobs_ = cacu_allocator::create_blobs();
	if (out_blob != NULL)
		blobs_->push_back(out_blob);
	add_op(operator_factory::create_op(op_, blobs_, _args));
	out_blob = _ops->back()->out_data<blob_base>();
	return this;
}

layer* layer::op(op_name op_, blob_base *blob_) {
	if (blob_ == NULL)
		LOG_FATAL("input data is NULL!");
	refresh_layer_param(blob_);
	blobs *blobs_ = cacu_allocator::create_blobs();
	if (out_blob != NULL)
		blobs_->push_back(out_blob);
	blobs_->push_back(blob_);
	add_op(operator_factory::create_op(op_, blobs_, _args));
	out_blob = _ops->back()->out_data<blob_base>();
	return this;
}

layer* layer::op(op_name op_, blob_base *blob_, data_args *args_) {
	if (blob_ == NULL)
		LOG_FATAL("input data is NULL!");
	_args = args_;
	blobs *blobs_ = cacu_allocator::create_blobs();
	if (out_blob != NULL)
		blobs_->push_back(out_blob);
	blobs_->push_back(blob_);
	add_op(operator_factory::create_op(op_, blobs_, _args));
	out_blob = _ops->back()->out_data<blob_base>();
	return this;
}

layer* layer::op(op_name op_, blobs *blobs_) {

	if (blobs_ == NULL)
		LOG_FATAL("input data is NULL!");
	refresh_layer_param(blobs_->at(0));

	if (out_blob != NULL)
		blobs_->push_back(out_blob);
	add_op(operator_factory::create_op(op_, blobs_, _args));
	out_blob = _ops->back()->out_data<blob_base>();
	return this;
}

layer* layer::op(op_name op_, blobs *blobs_, data_args *args_) {
	if (blobs_ == NULL)
		LOG_FATAL("input data is NULL!");
	_args = args_;
	if (out_blob != NULL)
		blobs_->push_back(out_blob);
	add_op(operator_factory::create_op(op_, blobs_, _args));
	out_blob = _ops->back()->out_data<blob_base>();
	return this;
}

}
