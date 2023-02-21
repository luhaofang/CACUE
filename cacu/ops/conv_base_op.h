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

#ifndef CONV_BASE_OP_H_
#define CONV_BASE_OP_H_

namespace cacu {

class conv_base_op: public operator_base {

public:

	//output_channel, kernel_size, stride, pad, input_dim, channel
	conv_base_op(blobs *&data, data_args *&args_, op_name type_) :
			operator_base(data, args_, type_) {
		_BASE_TYPE = CONV_BASE;
#if __USE_CUDNN__ == ON

		create_cudnn_handle(_conv_handle);
		create_convolution_desc(_conv_desc);
		create_tensor_4d_desc(_s_blob_desc);
		create_tensor_4d_desc(_o_blob_desc);

		_fwd_algo = (cudnnConvolutionFwdAlgo_t)0;
		_bwd_algo = (cudnnConvolutionBwdDataAlgo_t)0;
		_bwd_w_algo = (cudnnConvolutionBwdFilterAlgo_t)0;
#endif
	}

	~conv_base_op() {
#if __USE_CUDNN__ == ON
		destroy_convolution_descriptor(_conv_desc);
		destroy_tensor_descriptor(_s_blob_desc);
		destroy_tensor_descriptor(_o_blob_desc);
		release_cudnn_handle(_conv_handle);

#endif
	}

	void check() override {
		if(_args == NULL)
			LOG_FATAL("convolution data args cannot equal to NULL!");
		//output_channel > 0
		CHECK_GT_OP(_args->output_channel(), 0, "output_channel must > 0 vs %d",
				_args->output_channel());
		//kernel_size > 0
		CHECK_GT_OP(_args->kernel_size(), 0, "kernel_size must > 0 vs %d",
				_args->kernel_size());
		//stride > 0
		CHECK_GT_OP(_args->stride(), 0, "stride must > 0 vs %d",
				_args->stride());
		if(_args->size() > 5)
			_group = _args->at(6);
	}

	void load(std::ifstream& is) override {
		if (_w != NULL){
			if (_group != 1)
				_w->load_group(is, _group);
			else
				_w->load(is);
		}
		if(_bias != NULL){
			if (_is_use_bias)
				_bias->load(is);
		}
	}

	void save(std::ostream& os) override {
		if (_w != NULL){
			if (_group != 1)
				_w->serializa_group(os, _group);
			else
				_w->serializa(os);
		}
		if(_bias != NULL){
			if (_is_use_bias)
				_bias->serializa(os);
		}
	}

	inline void set_weight_init_type(param_init_type _type,
			float_t value = 0.0) {
		set_param_init_type(_type, _w, value);
	}

	inline void set_bias_init_type(param_init_type _type, float_t value = 0.0) {
		set_param_init_type(_type, _bias, value);
	}

	inline void set_group(int group) {
		CHECK_GT_OP(group, 0, "group must > 0 vs %d", group);
		CHECK_LE_OP(group, s_blobs->at(0)->channel(), "group must <= %d vs %d",
				_args->channel(), group);
		CHECK_EQ_OP(s_blobs->at(0)->channel() % group, 0,
				"channel mod group must == 0 vs %d", _args->channel() % group);
		LOG_INFO("group set: %d", group);
		this->_group = group;
		if(_args->size() <= 5)
			_args->push_back(_group);
		else
			_args->at(6) = _group;

#if __USE_CUDNN__ == ON
		_w->set_weight_desc(_w->num() / _group, _w->channel() / _group,
				_w->width(), _w->height());
		_bias->set_tensor_desc(1, _bias->num() / _group, 1, 1);

		initial();
#endif
	}

	void set_is_use_bias(bool switcher_) {
		_is_use_bias = switcher_;
	}

protected:

	bool _is_use_bias = true;

	weight *_w = NULL;
	weight *_bias = NULL;

	blob *_col_data = NULL;
	blob *_bias_multiplier = NULL;

	int _group = 1;
	int col_offset = 0;
	int w_offset = 0;
	int out_offset = 0;

#if __USE_CUDNN__ == ON

	cudnnHandle_t _conv_handle;

	cudnnConvolutionDescriptor_t _conv_desc;

	cudnnConvolutionFwdAlgo_t _fwd_algo;
	cudnnConvolutionBwdDataAlgo_t _bwd_algo;
	cudnnConvolutionBwdFilterAlgo_t _bwd_w_algo;

	cudnnTensorDescriptor_t _s_blob_desc;
	cudnnTensorDescriptor_t _o_blob_desc;

	size_t _fwd_workspace = 0;
	size_t _bwd_workspace = 0;
	size_t _bwd_w_workspace = 0;

	blob *_workspace_fwd = NULL;
	blob *_workspace_bwd = NULL;
	blob *_workspace_bwd_w = NULL;

#endif

};
}


#endif
