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
#include "../../utils/utils.h"

#include "operator_base.h"
#include "operators.h"

namespace mycnn{

	enum op_name
	{
		CACU_INNERPRODUCT,
		CACU_CONVOLUTION,
		CACU_SUM_ELEMWISE,
		CACU_RELU,
		CACU_BATCH_NORMALIZE,
		CACU_MAX_POOLING,
		CACU_AVERAGE_POOLING,
		CACU_DROPOUT,
		CACU_LEAKY_RELU
	};

	class operator_factory
	{
	
	public:	

		static operator_base *create_op(op_name op_name_, blobs *&blob_, args *&args_){

			switch (op_name_)
			{
			case CACU_INNERPRODUCT:
				CHECK_EQ_OP(blob_->size(), 1);
				return new inner_product_op((blob*&)(*blob_)[0], args_);
				break;
			case CACU_CONVOLUTION:
				CHECK_EQ_OP(blob_->size(), 1);
				return new convolution_op((blob*&)(*blob_)[0], args_);
				break;
			case CACU_SUM_ELEMWISE:
				CHECK_GE_OP(blob_->size(), 2);
				return new sum_elemwise_op(blob_, args_);
				break;
			case CACU_RELU:
				CHECK_EQ_OP(blob_->size(), 1);
				return new relu_op((blob*&)(*blob_)[0], args_);
				break;
			case CACU_BATCH_NORMALIZE:
				CHECK_EQ_OP(blob_->size(), 1);
				return new batch_normal_op((blob*&)(*blob_)[0], args_);
				break;
			case CACU_MAX_POOLING:
				CHECK_EQ_OP(blob_->size(), 1);
				return new max_pooling_op((blob*&)(*blob_)[0], args_);
				break;
			case CACU_AVERAGE_POOLING:
				CHECK_EQ_OP(blob_->size(), 1);
				return new average_pooling_op((blob*&)(*blob_)[0], args_);
				break;
			case CACU_DROPOUT:
				CHECK_EQ_OP(blob_->size(), 1);
				return new dropout_op((blob*&)(*blob_)[0], args_);
				break;
			case CACU_LEAKY_RELU:
				CHECK_EQ_OP(blob_->size(), 1);
				return new leaky_relu_op((blob*&)(*blob_)[0], args_);
				break;
			default:
				LOG_FATAL("No op is founded as: %d", op_name_);
				return NULL;
				break;
			}
			return NULL;
		}
	};

};
