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

#ifndef CUDNN_FUNCTIONS_H_
#define CUDNN_FUNCTIONS_H_

#include "../../config.h"
#include "cudnn_utils.h"

#if __USE_DEVICE__ == ON
#if __USE_CUDNN__ == ON

#include "cudnn.h"

namespace cacu {

inline cudnnDataType_t data_type() {
#if __CUDNN_DATA_TYPE__ == __FLOAT__
	return cudnnDataType_t::CUDNN_DATA_FLOAT;
#elif __CUDNN_DATA_TYPE__ == __DOUBLE__
	return cudnnDataType_t::CUDNN_DATA_DOUBLE;
#endif
}

inline void create_cudnn_handle(cudnnHandle_t &handle_) {
	CUDNN_CHECK(cudnnCreate(&handle_));
}

inline void release_cudnn_handle(cudnnHandle_t &handle_) {
	CUDNN_CHECK(cudnnDestroy(handle_));
}

inline void create_tensor_4d_desc(cudnnTensorDescriptor_t &data_desc_) {
	CUDNN_CHECK(cudnnCreateTensorDescriptor(&data_desc_));
}

inline void create_filter_desc(cudnnFilterDescriptor_t &filter_desc_) {
	CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
}

inline void create_convolution_desc(cudnnConvolutionDescriptor_t &conv_desc_) {
	CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
}

inline void create_pooling_desc(cudnnPoolingDescriptor_t &pooling_desc_) {
	CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc_));
}
inline void create_activation_desc(
		cudnnActivationDescriptor_t &activation_desc_) {
	CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc_));
}
inline void create_dropout_desc(cudnnDropoutDescriptor_t &dropout_desc_) {
	CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
}

inline void get_convolution_fwd_algo(cudnnHandle_t &handle_,
		cudnnConvolutionFwdAlgo_t &algo_, cudnnTensorDescriptor_t indata_desc_,
		cudnnFilterDescriptor_t filter_desc_,
		cudnnConvolutionDescriptor_t conv_desc_,
		cudnnTensorDescriptor_t outdata_desc_) {
#if __CUDNN_WORKSPACE_BALANCE__ == ON
	CUDNN_CHECK(
			cudnnGetConvolutionForwardAlgorithm(handle_, indata_desc_, filter_desc_, conv_desc_, outdata_desc_, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, WORKSPACE_LIMIT_BYTES, &algo_));
#else
	CUDNN_CHECK(
			cudnnGetConvolutionForwardAlgorithm(handle_, indata_desc_, filter_desc_, conv_desc_, outdata_desc_, CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, WORKSPACE_LIMIT_BYTES, &algo_));
#endif
}

inline void get_convolution_fwd_work_space(cudnnHandle_t &handle_,
		size_t &workspace_size, cudnnTensorDescriptor_t indata_desc_,
		cudnnFilterDescriptor_t filter_desc_,
		cudnnConvolutionDescriptor_t conv_desc_,
		cudnnTensorDescriptor_t outdata_desc_,
		cudnnConvolutionFwdAlgo_t algo_) {
	CUDNN_CHECK(
			cudnnGetConvolutionForwardWorkspaceSize(handle_, indata_desc_,
					filter_desc_, conv_desc_, outdata_desc_, algo_,
					&workspace_size));
}

inline void get_convolution_bwd_algo(cudnnHandle_t &handle_,
		cudnnConvolutionBwdDataAlgo_t &algo_,
		cudnnTensorDescriptor_t indiff_desc_,
		cudnnFilterDescriptor_t filter_desc_,
		cudnnConvolutionDescriptor_t conv_desc_,
		cudnnTensorDescriptor_t outdiff_desc_) {
#if __CUDNN_WORKSPACE_BALANCE__ == ON
	CUDNN_CHECK(
			cudnnGetConvolutionBackwardDataAlgorithm(handle_, filter_desc_, outdiff_desc_, conv_desc_, indiff_desc_, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, WORKSPACE_LIMIT_BYTES, &algo_));
#else
	CUDNN_CHECK(
			cudnnGetConvolutionBackwardDataAlgorithm(handle_, filter_desc_, outdiff_desc_, conv_desc_, indiff_desc_, CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, WORKSPACE_LIMIT_BYTES, &algo_));
#endif
}

inline void get_convolution_bwd_work_space(cudnnHandle_t &handle_,
		size_t &workspace_size, cudnnTensorDescriptor_t indiff_desc_,
		cudnnFilterDescriptor_t filter_desc_,
		cudnnConvolutionDescriptor_t conv_desc_,
		cudnnTensorDescriptor_t outdiff_desc_,
		cudnnConvolutionBwdDataAlgo_t algo_) {
	CUDNN_CHECK(
			cudnnGetConvolutionBackwardDataWorkspaceSize(handle_, filter_desc_,
					outdiff_desc_, conv_desc_, indiff_desc_, algo_,
					&workspace_size));
}

inline void get_convolution_bwd_filter_algo(cudnnHandle_t &handle_,
		cudnnConvolutionBwdFilterAlgo_t &algo_,
		cudnnTensorDescriptor_t indiff_desc_,
		cudnnFilterDescriptor_t filter_desc_,
		cudnnConvolutionDescriptor_t conv_desc_,
		cudnnTensorDescriptor_t outdiff_desc_) {
#if __CUDNN_WORKSPACE_BALANCE__ == ON
	CUDNN_CHECK(
			cudnnGetConvolutionBackwardFilterAlgorithm(handle_, indiff_desc_, outdiff_desc_, conv_desc_, filter_desc_, CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, WORKSPACE_LIMIT_BYTES, &algo_));
#else
	CUDNN_CHECK(
			cudnnGetConvolutionBackwardFilterAlgorithm(handle_, indiff_desc_, outdiff_desc_, conv_desc_, filter_desc_, CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, WORKSPACE_LIMIT_BYTES, &algo_));
#endif
}

inline void get_convolution_bwd_filter_work_space(cudnnHandle_t &handle_,
		size_t &workspace_size, cudnnTensorDescriptor_t indiff_desc_,
		cudnnFilterDescriptor_t filter_desc_,
		cudnnConvolutionDescriptor_t conv_desc_,
		cudnnTensorDescriptor_t outdiff_desc_,
		cudnnConvolutionBwdFilterAlgo_t algo_) {
	CUDNN_CHECK(
			cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_,
					indiff_desc_, outdiff_desc_, conv_desc_, filter_desc_,
					algo_, &workspace_size));
}

inline void set_tensor_4d_desc(cudnnTensorDescriptor_t &data_desc_, int num_,
		int channel_, int width_, int height_) {
//	LOG_DEBUG("set tensor_desc: %d, %d, %d, %d", num_, channel_, height_,
//			width_);
	CUDNN_CHECK(
			cudnnSetTensor4dDescriptor(data_desc_, CUDNN_TENSOR_NCHW,
					data_type(), num_, channel_, height_, width_));
}

inline void set_tensor_4d_desc(cudnnTensorDescriptor_t &data_desc_, int num_,
		int channel_, int width_, int height_, int stride_num_,
		int stride_channel_, int stride_width_, int stride_height_) {
//	LOG_DEBUG("set tensor_desc: %d, %d, %d, %d", num_, channel_, height_,
//			width_);
	CUDNN_CHECK(
			cudnnSetTensor4dDescriptor(data_desc_, CUDNN_TENSOR_NCHW,
					data_type(), num_, channel_, height_, width_));
}

inline void set_filter_4d_desc(cudnnFilterDescriptor_t &filter_desc_, int num_,
		int channel_, int width_, int height_) {
//	LOG_DEBUG("set filter_desc: %d, %d, %d, %d", num_, channel_, height_,
//			width_);
	CUDNN_CHECK(
			cudnnSetFilter4dDescriptor(filter_desc_, data_type(),
					CUDNN_TENSOR_NCHW, num_, channel_, height_, width_));
}

inline void set_convolution_desc(cudnnConvolutionDescriptor_t &conv_desc_,
		int pad_w_, int pad_h_, int stride_w_, int stride_h) {
//	LOG_DEBUG("set convolution_desc: %d, %d, %d, %d", pad_h_, pad_w_, stride_h,
//			stride_w_);
	CUDNN_CHECK(
			cudnnSetConvolution2dDescriptor(conv_desc_, pad_h_, pad_w_,
					stride_h, stride_w_, 1, 1, CUDNN_CROSS_CORRELATION,
					data_type()));
}

inline void set_pooling_desc(cudnnPoolingDescriptor_t &pooling_desc_,
		cudnnPoolingMode_t pooling_method_, int width_, int height_, int pad_w_,
		int pad_h_, int stride_w_, int stride_h_) {
//	LOG_DEBUG("set pooling_desc: %d, %d, %d, %d, %d, %d", height_, width_,
//			pad_h_, pad_w_, stride_h_, stride_w_);
	CUDNN_CHECK(
			cudnnSetPooling2dDescriptor(pooling_desc_, pooling_method_,
					CUDNN_PROPAGATE_NAN, height_, width_, pad_h_, pad_w_,
					stride_h_, stride_w_));
}

inline void set_activation_desc(cudnnActivationDescriptor_t &activation_desc_,
		cudnnActivationMode_t mode_, cudnnNanPropagation_t reluNanOpt_,
		double coef_) {
	CUDNN_CHECK(
			cudnnSetActivationDescriptor(activation_desc_, mode_, reluNanOpt_,
					coef_));
}

inline void destroy_filter_descriptor(cudnnFilterDescriptor_t &filter_desc_) {
	CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
}

inline void destroy_tensor_descriptor(cudnnTensorDescriptor_t &data_desc_) {
	CUDNN_CHECK(cudnnDestroyTensorDescriptor(data_desc_));
}

inline void destroy_convolution_descriptor(
		cudnnConvolutionDescriptor_t &conv_desc_) {
	CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

inline void destroy_pooling_descriptor(
		cudnnPoolingDescriptor_t &pooling_desc_) {
	CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pooling_desc_));
}

inline void destroy_activation_descriptor(
		cudnnActivationDescriptor_t &activation_desc_) {
	CUDNN_CHECK(cudnnDestroyActivationDescriptor(activation_desc_));
}

inline void destroy_dropout_descriptor(
		cudnnDropoutDescriptor_t &dropout_desc_) {
	CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
}

inline void cudnn_convolution_forward_data(cudnnHandle_t &handle_,
		cudnnTensorDescriptor_t indata_desc_, float_t *indata_,
		cudnnFilterDescriptor_t filter_desc_, float_t *w_,
		cudnnConvolutionDescriptor_t conv_desc_,
		cudnnTensorDescriptor_t outdata_desc_, float_t *outdata_,
		cudnnConvolutionFwdAlgo_t fwd_algo_, float_t *workspace,
		size_t workspace_size) {
	float_t one = 1;
	float_t zero = 0;
	CUDNN_CHECK(
			cudnnConvolutionForward(handle_, &one, indata_desc_, indata_,
					filter_desc_, w_, conv_desc_, fwd_algo_, workspace,
					workspace_size, &zero, outdata_desc_, outdata_));
}

inline void cudnn_convolution_forward_bias(cudnnHandle_t &handle_,
		cudnnTensorDescriptor_t bias_desc_, float_t *bias_,
		cudnnTensorDescriptor_t outdata_desc_, float_t *outdata_) {
	float_t one = 1;
	CUDNN_CHECK(
			cudnnAddTensor(handle_, &one, bias_desc_, bias_, &one,
					outdata_desc_, outdata_));
}

inline void cudnn_convolution_backward_data(cudnnHandle_t &handle_,
		cudnnTensorDescriptor_t indiff_desc_, float_t *indiff_,
		cudnnFilterDescriptor_t filter_desc_, float_t *w_,
		cudnnConvolutionDescriptor_t conv_desc_,
		cudnnTensorDescriptor_t outdiff_desc_, float_t *outdiff_,
		cudnnConvolutionBwdDataAlgo_t bwd_algo_, float_t *workspace,
		size_t workspace_size) {
	float_t one = 1;
	float_t zero = 0;
	CUDNN_CHECK(
			cudnnConvolutionBackwardData(handle_, &one, filter_desc_, w_,
					outdiff_desc_, outdiff_, conv_desc_, bwd_algo_, workspace,
					workspace_size, &zero, indiff_desc_, indiff_));
}

inline void cudnn_convolution_backward_weight(cudnnHandle_t &handle_,
		cudnnTensorDescriptor_t indata_desc_, float_t *indata_,
		cudnnFilterDescriptor_t filter_desc_, float_t *w_diff_,
		cudnnConvolutionDescriptor_t conv_desc_,
		cudnnTensorDescriptor_t outdiff_desc_, float_t *outdiff_,
		cudnnConvolutionBwdFilterAlgo_t bwd_w_algo_, float_t *workspace,
		size_t workspace_size) {
	float_t one = 1;
	CUDNN_CHECK(
			cudnnConvolutionBackwardFilter(handle_, &one, indata_desc_, indata_,
					outdiff_desc_, outdiff_, conv_desc_, bwd_w_algo_, workspace,
					workspace_size, &one, filter_desc_, w_diff_));
}

inline void cudnn_convolution_backward_bias(cudnnHandle_t &handle_,
		cudnnTensorDescriptor_t bias_desc_, float_t *bias_diff_,
		cudnnTensorDescriptor_t outdiff_desc_, float_t *outdiff_) {
	float_t one = 1;
	CUDNN_CHECK(
			cudnnConvolutionBackwardBias(handle_, &one, outdiff_desc_, outdiff_,
					&one, bias_desc_, bias_diff_));
}

inline void cudnn_pooling_forward(cudnnHandle_t &handle_,
		cudnnTensorDescriptor_t indata_desc_, float_t *indata_,
		cudnnPoolingDescriptor_t pooling_desc,
		cudnnTensorDescriptor_t outdata_desc_, float_t *outdata_) {
	float_t one = 1;
	float_t zero = 0;
	CUDNN_CHECK(
			cudnnPoolingForward(handle_, pooling_desc, &one, indata_desc_,
					indata_, &zero, outdata_desc_, outdata_));
}

inline void cudnn_pooling_backward(cudnnHandle_t &handle_,
		cudnnTensorDescriptor_t indata_desc_, float_t *indata_,
		cudnnTensorDescriptor_t indiff_desc_, float_t *indiff_,
		cudnnPoolingDescriptor_t pooling_desc,
		cudnnTensorDescriptor_t outdata_desc_, float_t *outdata_,
		cudnnTensorDescriptor_t outdiff_desc_, float_t *outdiff_) {
	float_t one = 1;
	float_t zero = 0;
	CUDNN_CHECK(
			cudnnPoolingBackward(handle_, pooling_desc, &one, outdata_desc_,
					outdata_, outdiff_desc_, outdiff_, indata_desc_, indata_,
					&zero, indiff_desc_, indiff_));
}

inline void cudnn_bn_training_forward(cudnnHandle_t &handle_,
		cudnnTensorDescriptor_t indata_desc_, float_t *indata_,
		cudnnBatchNormMode_t mode,
		cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_, float_t* bnScale_,
		float_t* bnBias_, float_t exponentialAverageFactor_,
		float_t* resultRunningMean_, float_t* resultRunningVariance_,
		float_t epsilon_, float_t* resultSaveMean_,
		float_t* resultSaveInvVariance_, cudnnTensorDescriptor_t outdata_desc_,
		float_t *outdata_) {
	float_t one = 1;
	float_t zero = 0;
	CUDNN_CHECK(
			cudnnBatchNormalizationForwardTraining(handle_, mode, &one, &zero,
					indata_desc_, indata_, outdata_desc_, outdata_,
					bnScaleBiasMeanVarDesc_, bnScale_, bnBias_,
					exponentialAverageFactor_, resultRunningMean_,
					resultRunningVariance_, epsilon_, resultSaveMean_,
					resultSaveInvVariance_));
}

inline void cudnn_bn_inference_forward(cudnnHandle_t &handle_,
		cudnnTensorDescriptor_t indata_desc_, float_t *indata_,
		cudnnBatchNormMode_t mode,
		cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_, float_t* bnScale_,
		float_t* bnBias_, float_t* estimatedMean_, float_t* estimatedVariance_,
		float_t epsilon_, cudnnTensorDescriptor_t outdata_desc_,
		float_t *outdata_) {
	float_t one = 1;
	float_t zero = 0;
	CUDNN_CHECK(
			cudnnBatchNormalizationForwardInference(handle_, mode, &one, &zero,
					indata_desc_, indata_, outdata_desc_, outdata_,
					bnScaleBiasMeanVarDesc_, bnScale_, bnBias_, estimatedMean_,
					estimatedVariance_, epsilon_));
}

inline void cudnn_bn_backward(cudnnHandle_t &handle_,
		cudnnTensorDescriptor_t indata_desc_, float_t *indata_,
		cudnnBatchNormMode_t mode, cudnnTensorDescriptor_t bnScaleBiasDiffDesc_,
		cudnnTensorDescriptor_t indiff_desc_, float_t *indiff_,
		float_t* bnScale_, float_t* resultBnScaleDiff_,
		float_t* resultBnBiasDiff_, float_t epsilon_, float_t* savedMean_,
		float_t* savedInvVariance_, cudnnTensorDescriptor_t outdata_desc_,
		float_t *outdata_, cudnnTensorDescriptor_t outdiff_desc_,
		float_t *outdiff_) {
	float_t one = 1;
	float_t zero = 0;
	CUDNN_CHECK(
			cudnnBatchNormalizationBackward(handle_, mode, &one, &zero, &one,
					&one, indata_desc_, indata_, outdiff_desc_, outdiff_,
					indiff_desc_, indiff_, bnScaleBiasDiffDesc_, bnScale_,
					resultBnScaleDiff_, resultBnBiasDiff_, epsilon_, savedMean_,
					savedInvVariance_));
}

inline void cudnn_activation_forward(cudnnHandle_t &handle_,
		cudnnActivationDescriptor_t activation_desc_,
		cudnnTensorDescriptor_t indata_desc_, float_t *indata_,
		cudnnTensorDescriptor_t outdata_desc_, float_t *outdata_) {
	float_t one = 1;
	float_t zero = 0;
	cudnnActivationForward(handle_, activation_desc_, &one, indata_desc_,
			indata_, &zero, outdata_desc_, outdata_);
}

inline void cudnn_activation_backward(cudnnHandle_t &handle_,
		cudnnActivationDescriptor_t activation_desc_,
		cudnnTensorDescriptor_t indata_desc_, float_t *indata_,
		cudnnTensorDescriptor_t indiff_desc_, float_t *indiff_,
		cudnnTensorDescriptor_t outdata_desc_, float_t *outdata_,
		cudnnTensorDescriptor_t outdiff_desc_, float_t *outdiff_) {
	float_t one = 1;
	float_t zero = 0;
	cudnnActivationBackward(handle_, activation_desc_, &one, outdata_desc_,
			outdata_, outdiff_desc_, outdiff_, indata_desc_, indata_, &zero,
			indiff_desc_, indiff_);
}

inline void cudnn_dropout_forward(cudnnHandle_t handle_,
		cudnnDropoutDescriptor_t dropout_desc_,
		cudnnTensorDescriptor_t indata_desc_, float_t *indata_,
		cudnnTensorDescriptor_t outdata_desc_, float_t *outdata_,
		float_t *reserve_space_, size_t reserve_space_size_in_bytes_) {
	cudnnDropoutForward(handle_, dropout_desc_, indata_desc_, indata_,
			outdata_desc_, outdata_, reserve_space_,
			reserve_space_size_in_bytes_);
}

inline void cudnn_dropout_backward(cudnnHandle_t handle_,
		cudnnDropoutDescriptor_t dropout_desc_,
		cudnnTensorDescriptor_t outdiff_desc_, float_t *outdiff_,
		cudnnTensorDescriptor_t indiff_desc_, float_t *indiff_,
		float_t *reserve_space_, size_t reserve_space_size_in_bytes_) {
	cudnnDropoutBackward(handle_, dropout_desc_, outdiff_desc_, outdiff_,
			indiff_desc_, indiff_, reserve_space_,
			reserve_space_size_in_bytes_);
}

}

#endif
#endif

#endif
