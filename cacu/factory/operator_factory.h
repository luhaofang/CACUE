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

#ifndef OPERATOR_FACTORY_H_
#define OPERATOR_FACTORY_H_

#include "../framework/blob.h"
#include "../framework/blobs.h"
#include "../framework/data_args.h"

#include "../ops/operator_base.h"
#include "../ops/operators.h"
#include "../ops/ops_definition.h"

namespace cacu {

class operator_factory {

public:

	static operator_base* const create_op(op_name op_name_, blobs *blob_,
			data_args *d_args_, op_args *o_args_) {

		switch (op_name_) {
		case CACU_INNERPRODUCT:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new inner_product_op(blob_->at(0), d_args_);
		case CACU_CONVOLUTION:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new convolution_op(blob_->at(0), d_args_);
		case CACU_SUM_ELEMWISE:
			CHECK_GE_OP(blob_->size(), 2, "blobs size must >= 2 vs %d",
					blob_->size())
			;
			return new sum_elemwise_op(blob_, d_args_);
		case CACU_RELU:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new relu_op(blob_->at(0));
		case CACU_BATCH_NORMALIZE:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new batch_normalize_op(blob_->at(0), d_args_);
		case CACU_MAX_POOLING:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new max_pooling_op(blob_->at(0), d_args_);
		case CACU_AVERAGE_POOLING:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new average_pooling_op(blob_->at(0), d_args_);
		case CACU_DROPOUT:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new dropout_op(blob_->at(0), d_args_);
		case CACU_LEAKY_RELU:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new leaky_relu_op(blob_->at(0));
		case CACU_SOFTMAX:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new softmax_op(blob_->at(0));
		case CACU_SOFTMAX_LOSS:
			CHECK_EQ_OP(blob_->size(), 2, "blobs size must == 2 vs %d",
					blob_->size())
			;
			return new softmax_with_loss_op(blob_, d_args_);

		case CACU_SIGMOID_LOSS:
			CHECK_EQ_OP(blob_->size(), 2, "blobs size must == 2 vs %d",
					blob_->size())
			;
			return new sigmoid_with_loss_op(blob_, d_args_);
		case CACU_MSE_LOSS:
			CHECK_EQ_OP(blob_->size(), 2, "blobs size must == 2 vs %d",
					blob_->size())
			;
			return new mse_loss_op(blob_);
		case CACU_FEATURE_COMBINE:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new feature_combine_op(blob_->at(0), d_args_);
		case CACU_SPLIT:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new split_op(blob_->at(0), o_args_);
		case CACU_PRELU:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new prelu_op(blob_->at(0));
		case CACU_ROI_POOLING:
			CHECK_EQ_OP(blob_->size(), 2, "blobs size must == 2 vs %d",
					blob_->size())
			;
			return new roi_pooling_op(blob_, d_args_);
		case CACU_P_INNERPRODUCT:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new p_inner_product_op(blob_->at(0), d_args_);
		case CACU_DECONVOLUTION:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new deconvolution_op(blob_->at(0), d_args_);
		case CACU_INJECTOR:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new injector_op(blob_->at(0));
		case CACU_ROW_MAX_POOLING:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new row_max_pooling_op(blob_->at(0), d_args_);
		case CACU_TANH:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new tanh_op(blob_->at(0));
		case CACU_SELU:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new selu_op(blob_->at(0));
		case CACU_HTANH:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
				blob_->size())
				;
			return new half_tanh_op(blob_->at(0));
		case CACU_SIGMOID:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new sigmoid_op(blob_->at(0));
		case CACU_PRIMARY_VECTOR:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new primary_vector_op(blob_->at(0), d_args_);
		case CACU_MATH:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
				blob_->size())
				;
			return new math_op(blob_->at(0));
		case CACU_DILATED_CONVOLUTION:
			CHECK_EQ_OP(blob_->size(), 1, "blobs size must == 1 vs %d",
					blob_->size())
			;
			return new dilated_convolution_op(blob_->at(0), d_args_);
		case CACU_COMBINATION:
			CHECK_GE_OP(blob_->size(), 2, "blobs size must >= 2 vs %d",
					blob_->size())
			;
			return new combination_op(blob_, d_args_);
		default:
			LOG_FATAL("No op is founded as: %d", op_name_)
			;
			return NULL;
		}
		//*/
		return NULL;
	}

	/*
	template<class OPTYPE>
	static OPTYPE*& trans_op(operator_base* const op_) {
		/*
		 switch (op_->_TYPE()) {
		 case CACU_INNERPRODUCT:
		 return (inner_product_op*&) op_;
		 case CACU_CONVOLUTION:
		 return (convolution_op*&) op_;
		 case CACU_SUM_ELEMWISE:
		 return (sum_elemwise_op*&) op_;
		 case CACU_RELU:
		 return (relu_op*&) op_;
		 case CACU_BATCH_NORMALIZE:
		 return (batch_normalize_op*&) op_;
		 case CACU_MAX_POOLING:
		 return (max_pooling_op*&) op_;
		 case CACU_AVERAGE_POOLING:
		 return (average_pooling_op*&) op_;
		 case CACU_DROPOUT:
		 return (dropout_op*&) op_;
		 case CACU_LEAKY_RELU:
		 return (leaky_relu_op*&) op_;
		 case CACU_SOFTMAX:
		 return (softmax_op*&) op_;
		 case CACU_SOFTMAX_LOSS:
		 return (softmax_with_loss_op*&) op_;
		 case CACU_FEATURE_COMBINE:
		 return (feature_combine_op*&) op_;
		 case CACU_SPLIT:
		 return (split_op*&) op_;
		 case CACU_HINGE_LOSS:
		 return (hinge_loss_op*&) op_;
		 case CACU_ROI_POOLING:
		 return (roi_pooling_op*&) op_;
		 case CACU_P_INNERPRODUCT:
		 return (p_inner_product_op*&) op_;
		 case CACU_DECONVOLUTION:
		 return (deconvolution_op*&) op_;
		 case CACU_INJECTOR:
		 return (injector_op*&) op_;
		 case CACU_ROW_MAX_POOLING:
		 return (row_max_pooling_op*&) op_;
		 case CACU_TANH:
		 return (tanh_op*&) op_;
		 case CACU_SIGMOID:
		 return (sigmoid_op*&) op_;
		 case CACU_PRIMARY_VECTOR:
		 return (primary_vector_op*&) op_;
		 default:
		 LOG_FATAL("No op is founded as: %d", op_->_TYPE())
		 ;
		 return NULL;
		 }

		return NULL;
	}
	*/
};

}


#endif
