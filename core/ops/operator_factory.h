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

#pragma once

#include "../framework/blob.h"
#include "../utils/utils.h"

#include "operator_base.h"
#include "operators.h"

namespace mycnn{

	enum op_name
	{
		CACU_INNERPRODUCT,
		CACU_CONVOLUTION,
		CACU_BATCH_NORMALIZE,
		CACU_DROPOUT,
		CACU_P_INNERPRODUCT,
		CACU_DECONVOLUTION,

		CACU_MAX_POOLING,
		CACU_AVERAGE_POOLING,
		CACU_ROI_POOLING,

		CACU_RELU,
		CACU_LEAKY_RELU,
		CACU_SOFTMAX,

		CACU_SOFTMAX_LOSS,
		CACU_HINGE_LOSS,

		CACU_SUM_ELEMWISE,
		CACU_FEATURE_COMBINE,
		CACU_SPLIT
	};

	class operator_factory
	{

	public:	

		static operator_base* const create_op(op_name op_name_, blobs *&blob_, args *&args_){

			switch (op_name_)
			{
			case CACU_INNERPRODUCT:
				CHECK_EQ_OP(blob_->size(), 1 , "blobs size must == 1 vs %d",blob_->size());
				return new inner_product_op(blob_->at(0), args_);
			case CACU_CONVOLUTION:
				CHECK_EQ_OP(blob_->size(), 1 , "blobs size must == 1 vs %d",blob_->size());
				return new convolution_op(blob_->at(0), args_);
			case CACU_SUM_ELEMWISE:
				CHECK_GE_OP(blob_->size(), 2 , "blobs size must >= 2 vs %d",blob_->size());
				return new sum_elemwise_op(blob_, args_);
			case CACU_RELU:
				CHECK_EQ_OP(blob_->size(), 1 , "blobs size must == 1 vs %d",blob_->size());
				return new relu_op(blob_->at(0), args_);
			case CACU_BATCH_NORMALIZE:
				CHECK_EQ_OP(blob_->size(), 1 , "blobs size must == 1 vs %d",blob_->size());
				return new batch_normal_op(blob_->at(0), args_);
			case CACU_MAX_POOLING:
				CHECK_EQ_OP(blob_->size(), 1 , "blobs size must == 1 vs %d",blob_->size());
				return new max_pooling_op(blob_->at(0), args_);
			case CACU_AVERAGE_POOLING:
				CHECK_EQ_OP(blob_->size(), 1 , "blobs size must == 1 vs %d",blob_->size());
				return new average_pooling_op(blob_->at(0), args_);
			case CACU_DROPOUT:
				CHECK_EQ_OP(blob_->size(), 1 , "blobs size must == 1 vs %d",blob_->size());
				return new dropout_op(blob_->at(0), args_);
			case CACU_LEAKY_RELU:
				CHECK_EQ_OP(blob_->size(), 1 , "blobs size must == 1 vs %d",blob_->size());
				return new leaky_relu_op(blob_->at(0), args_);
			case CACU_SOFTMAX:
				CHECK_EQ_OP(blob_->size(), 1 , "blobs size must == 1 vs %d",blob_->size());
				return new softmax_op(blob_->at(0), args_);
			case CACU_SOFTMAX_LOSS:
				CHECK_EQ_OP(blob_->size(), 2 , "blobs size must == 2 vs %d",blob_->size());
				return new softmax_with_loss_op(blob_, args_);
			case CACU_FEATURE_COMBINE:
				CHECK_EQ_OP(blob_->size(), 1 , "blobs size must == 1 vs %d",blob_->size());
				return new feature_combine_op(blob_->at(0), args_);
			case CACU_SPLIT:
				CHECK_EQ_OP(blob_->size(), 1 , "blobs size must == 1 vs %d",blob_->size());
				return new split_op(blob_->at(0), args_);
			case CACU_HINGE_LOSS:
				CHECK_EQ_OP(blob_->size(), 2 , "blobs size must == 2 vs %d",blob_->size());
				return new hinge_loss_op(blob_, args_);
			case CACU_ROI_POOLING:
				CHECK_EQ_OP(blob_->size(), 2 , "blobs size must == 2 vs %d",blob_->size());
				return new roi_pooling_op(blob_, args_);
			case CACU_P_INNERPRODUCT:
				CHECK_EQ_OP(blob_->size(), 1 , "blobs size must == 1 vs %d",blob_->size());
				return new p_innerproduct_op(blob_->at(0), args_);
			case CACU_DECONVOLUTION:
				CHECK_EQ_OP(blob_->size(), 1 , "blobs size must == 1 vs %d",blob_->size());
				return new deconvolution_op(blob_->at(0), args_);
			default:
				LOG_FATAL("No op is founded as: %d", op_name_);
				return NULL;
			}
			return NULL;
		}
	};

};
