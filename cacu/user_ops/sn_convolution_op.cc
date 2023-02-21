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

#include "sn_convolution_op.h"

#include "../math/math.h"

namespace cacu {

	//output_channel, kernel_size, stride, pad, input_dim, channel
	sn_convolution_op::sn_convolution_op(blobs *&data, data_args *&args_) :
			convolution_op(data, args_) {
		_OP_TYPE = CACU_SN_CONVOLUTION;
		init_weights();
		convolution_op::set_is_use_bias(false);
	}

	sn_convolution_op::~sn_convolution_op() {

		delete _u;
		delete _v;
		delete _w_original;
	}

	void sn_convolution_op::init_weights() {
		weight *w_ = get_weight(0);

		LOG_INFO("Initalize the parameters for spectral normalization!");
		_u = new blob(w_ ->num(), 1, 1, 1, 0, train);
		_v = new blob(1, w_->channel(), w_->width(), w_->height(), 0, test);
		_w_original = new blob(w_ ->num(), w_->channel(), w_->width(), w_->height(), 0, test);
		_u->set_init_type(gaussian, 1);

	}

	void sn_convolution_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {
		if(_is_norm)
			norm_weight(get_weight(0));
		convolution_op::op(s_blobs_, o_blobs_);
	}

	void sn_convolution_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {
		convolution_op::grad(s_blobs_, o_blobs_);
		if(_is_norm)
			norm_recover(get_weight(0));
	}

	void sn_convolution_op::echo()
	{
		LOG_INFO(
				"sn convolution op: in(%d, %d, %d), out(%d, %d, %d), kernel/stride/pad([%d, %d], %d, %d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height(),
				_args->kernel_size(), _args->kernel_size(), _args->stride(), _args->pad());
	}

	void sn_convolution_op::norm_weight(weight *&w_) {

		cacu_copy(w_->s_data(), w_->count(), _w_original->s_data());
		//_u->set_init_type(gaussian, 1);
		for (int i = 0; i < _pow_iter; ++i){
			//cacu_sgemm(NOTRANS,TRANS,_u->s_data(),1,_u->count(),w_->s_data(),w_->length(),(float_t)1.,_v->s_data(),(float_t)0);
			cacu_sgemv(NOTRANS, w_->s_data(), w_->length(), _u->s_data(), w_->num(), (float_t)1., _v->s_data(), (float_t)0);
			cacu_l2_normalization(_v->s_data(),_v->count(), 1e-12);

			//cacu_sgemm(NOTRANS,NOTRANS,_v->s_data(),1,_v->count(),w_->s_data(),w_->num(),(float_t)1.,_u->s_data(),(float_t)0);
			cacu_sgemv(TRANS, w_->s_data(), w_->length(), _v->s_data(), w_->num(), (float_t)1., _u->s_data(), (float_t)0);
			cacu_l2_normalization(_u->s_data(),_u->count(), 1e-12);
		}
		//cacu_print("u", _u->s_data(), _u->count());
		//cacu_sgemm(NOTRANS,NOTRANS,_v->s_data(),1,_v->count(),w_->s_data(),w_->num(),(float_t)1.,_u->s_diff(),(float_t)0);
		cacu_sgemv(TRANS, w_->s_data(), w_->length(), _v->s_data(), w_->num(), (float_t)1.,_u->s_diff(),(float_t)0);
		//cacu_sgemm(NOTRANS,NOTRANS,_u->s_diff(),1,_u->count(),_u->s_data(),1, (float_t)1.,_v->s_data(),(float_t)0);
		cacu_sgemv(NOTRANS, _u->s_diff(), 1, _u->s_data(), w_->num(), (float_t)1., _v->s_data(), (float_t)0);
		//_ws->switch_channel();
		//cacu_print("gamma", _v->s_data(), 1);
		//float_t _sigma = 0.;
	#if __USE_DEVICE__
		device_copy2host(&_sigma, _v->s_data(), 1);
	#else
		_sigma = _v->s_data()[0];
	#endif
		cacu_scalex(w_->s_data(), w_->count(), (float_t)1. / _sigma);
		//cacu_print(s_data(), 100);
	}

	void sn_convolution_op::load(std::ifstream& is) {
		convolution_op::load(is);
		_u->load(is);
	}

	void sn_convolution_op::save(std::ostream& os) {
		convolution_op::save(os);
		_u->serializa(os);
	}

	void sn_convolution_op::norm_recover(weight *&w_) {
		cacu_copy(_w_original->s_data(), w_->count(), w_->s_data());
//		cacu_scalex(w_->s_diff(), w_->count(), (float_t)1. / _sigma);
	}


REGISTER_OP_CLASS_D_DP(sn_convolution_op);
REGISTER_OP_MACRO(sn_convolution_op, CACU_SN_CONVOLUTION);

}
