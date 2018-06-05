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

#ifndef MODELS_CREATER_H_
#define MODELS_CREATER_H_

#include "../../cacu/cacu.h"
#include "../../cacu/framework/network.h"
#include "../../cacu/framework/layer.h"

using namespace cacu;

network* create_Pnet(dsize_t batch_size, phase_type phase_) {
	blobs *input_datas_ = cacu_allocator::create_blobs();
	blob *blob_ = cacu_allocator::create_blob(batch_size, 3, 12, 12, phase_);
	input_datas_->push_back(blob_);
	bin_blob *label_;
	blob *roi_label_ ;

	if (train == phase_) {
		label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1,
				phase_);
		roi_label_ = cacu_allocator::create_blob(batch_size, 4, 1, 1,
						phase_);
		input_datas_->push_back(label_);
		input_datas_->push_back(roi_label_);
	}

	network *net = new network(input_datas_);

	layer *conv1 = new layer(new data_args(10, 3, 1, 0, blob_->channel()));
	conv1->op(CACU_CONVOLUTION, blob_)->op(CACU_PRELU);
	*net << conv1;
	layer *pool = new layer(new data_args(10, 2, 2, 0, conv1->get_oblob()->channel()));
	pool->op(CACU_MAX_POOLING, conv1->get_oblob());
	*net << pool;
	layer *conv2 = new layer(new data_args(16, 3, 1, 0, pool->get_oblob()->channel()));
	conv2->op(CACU_CONVOLUTION, pool->get_oblob())->op(CACU_PRELU);
	*net << conv2;
	layer *conv3 = new layer(new data_args(32, 3, 1, 0, conv2->get_oblob()->channel()));
	conv3->op(CACU_CONVOLUTION, conv2->get_oblob())->op(CACU_PRELU);
	*net << conv3;
	layer *split = new layer();
	split->op(CACU_SPLIT,conv3->get_oblob(), new op_args(2));
	*net << split;
	layer *conv4_1 = new layer(new data_args(2, 1, 1, 0, split->get_oblobs()->at(0)->channel()));
	conv4_1->op(CACU_CONVOLUTION,  split->get_oblobs()->at(0));
	*net << conv4_1;
	layer *conv4_2 = new layer(new data_args(4, 1, 1, 0, split->get_oblobs()->at(1)->channel()));
	conv4_2->op(CACU_CONVOLUTION,  split->get_oblobs()->at(1));
	*net << conv4_2;

	layer *cls = new layer();
	if(train == phase_){
		blobs *cls_blobs = new blobs();
		cls_blobs->push_back(conv4_1->get_oblob());
		cls_blobs->push_back(label_);
		cls->op(CACU_SOFTMAX_LOSS,cls_blobs);
		cls->get_op<softmax_with_loss_op>(0, CACU_SOFTMAX_LOSS)->set_loss_weight(0.7);
	}
	else
		cls->op(CACU_SOFTMAX,conv4_1->get_oblob());
	*net << cls;

	layer *roi = new layer();
	if(train == phase_){
		blobs *roi_blobs = new blobs();
		roi_blobs->push_back(conv4_2->get_oblob());
		roi_blobs->push_back(roi_label_);
		roi->op(CACU_MSE_LOSS,roi_blobs);
		roi->get_op<mse_loss_op>(0, CACU_SOFTMAX_LOSS)->set_loss_weight(0.3);
		*net << roi;
	}

	return net;
}


network* create_Rnet(dsize_t batch_size, phase_type phase_) {
	blobs *input_datas_ = cacu_allocator::create_blobs();
	blob *blob_ = cacu_allocator::create_blob(batch_size, 3, 24, 24, phase_);
	input_datas_->push_back(blob_);
	bin_blob *label_;
	blob *roi_label_ ;

	if (train == phase_) {
		label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1,
				phase_);
		roi_label_ = cacu_allocator::create_blob(batch_size, 4, 1, 1,
						phase_);
		input_datas_->push_back(label_);
		input_datas_->push_back(roi_label_);
	}

	network *net = new network(input_datas_);

	layer *conv1 = new layer(new data_args(28, 3, 1, 0, blob_->channel()));
	conv1->op(CACU_CONVOLUTION, blob_)->op(CACU_PRELU);
	*net << conv1;
	layer *pool1 = new layer(new data_args(28, 3, 2, 0, conv1->get_oblob()->channel()));
	pool1->op(CACU_MAX_POOLING, conv1->get_oblob());
	*net << pool1;
	layer *conv2 = new layer(new data_args(48, 3, 1, 0, pool1->get_oblob()->channel()));
	conv2->op(CACU_CONVOLUTION, pool1->get_oblob())->op(CACU_PRELU);
	*net << conv2;
	layer *pool2 = new layer(new data_args(48, 3, 2, 0, conv2->get_oblob()->channel()));
	pool2->op(CACU_MAX_POOLING, conv2->get_oblob());
	*net << pool2;
	layer *conv3 = new layer(new data_args(64, 2, 1, 0, pool2->get_oblob()->channel()));
	conv3->op(CACU_CONVOLUTION, pool2->get_oblob())->op(CACU_PRELU);
	*net << conv3;
	layer *conv_fc = new layer(new data_args(128, 3, 1, 0, conv3->get_oblob()->channel()));
	conv_fc->op(CACU_CONVOLUTION, conv3->get_oblob())->op(CACU_PRELU);
	*net << conv_fc;
	layer *split = new layer();
	split->op(CACU_SPLIT,conv_fc->get_oblob(), new op_args(2));
	*net << split;
	layer *conv4_1 = new layer(new data_args(2, 1, 1, 0, split->get_oblobs()->at(0)->channel()));
	conv4_1->op(CACU_CONVOLUTION,  split->get_oblobs()->at(0));
	*net << conv4_1;
	layer *conv4_2 = new layer(new data_args(4, 1, 1, 0, split->get_oblobs()->at(1)->channel()));
	conv4_2->op(CACU_CONVOLUTION,  split->get_oblobs()->at(1));
	*net << conv4_2;

	layer *cls = new layer();
	if(train == phase_){
		blobs *cls_blobs = new blobs();
		cls_blobs->push_back(conv4_1->get_oblob());
		cls_blobs->push_back(label_);
		cls->op(CACU_SOFTMAX_LOSS,cls_blobs);
		cls->get_op<softmax_with_loss_op>(0, CACU_SOFTMAX_LOSS)->set_loss_weight(0.7);
	}
	else
		cls->op(CACU_SOFTMAX,conv4_1->get_oblob());
	*net << cls;

	layer *roi = new layer();
	if(train == phase_){
		blobs *roi_blobs = new blobs();
		roi_blobs->push_back(conv4_2->get_oblob());
		roi_blobs->push_back(roi_label_);
		roi->op(CACU_MSE_LOSS,roi_blobs);
		roi->get_op<mse_loss_op>(0, CACU_MSE_LOSS)->set_loss_weight(0.3);
		*net << roi;
	}


	return net;
}

network* create_Onet(dsize_t batch_size, phase_type phase_) {
	blobs *input_datas_ = cacu_allocator::create_blobs();
	blob *blob_ = cacu_allocator::create_blob(batch_size, 3, 48, 48, phase_);
	input_datas_->push_back(blob_);
	bin_blob *label_;
	blob *roi_label_ ;

	if (train == phase_) {
		label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1,
				phase_);
		roi_label_ = cacu_allocator::create_blob(batch_size, 4, 1, 1,
						phase_);
		input_datas_->push_back(label_);
		input_datas_->push_back(roi_label_);
	}

	network *net = new network(input_datas_);

	layer *conv1 = new layer(new data_args(32, 3, 1, 0, blob_->channel()));
	conv1->op(CACU_CONVOLUTION, blob_)->op(CACU_PRELU);
	*net << conv1;
	layer *pool1 = new layer(new data_args(32, 3, 2, 0, conv1->get_oblob()->channel()));
	pool1->op(CACU_MAX_POOLING, conv1->get_oblob());
	*net << pool1;
	layer *conv2 = new layer(new data_args(64, 3, 1, 0, pool1->get_oblob()->channel()));
	conv2->op(CACU_CONVOLUTION, pool1->get_oblob())->op(CACU_PRELU);
	*net << conv2;
	layer *pool2 = new layer(new data_args(64, 3, 2, 0, conv2->get_oblob()->channel()));
	pool2->op(CACU_MAX_POOLING, conv2->get_oblob());
	*net << pool2;
	layer *conv3 = new layer(new data_args(64, 3, 1, 0, pool2->get_oblob()->channel()));
	conv3->op(CACU_CONVOLUTION, pool2->get_oblob())->op(CACU_PRELU);
	*net << conv3;
	layer *pool3 = new layer(new data_args(64, 2, 2, 0, conv3->get_oblob()->channel()));
	pool3->op(CACU_MAX_POOLING, conv3->get_oblob());
	*net << pool3;
	layer *conv4 = new layer(new data_args(128, 2, 1, 0, pool3->get_oblob()->channel()));
	conv4->op(CACU_CONVOLUTION, pool3->get_oblob())->op(CACU_PRELU);
	*net << conv4;
	layer *conv_fc = new layer(new data_args(256, 3, 1, 0, conv4->get_oblob()->channel()));
	conv_fc->op(CACU_CONVOLUTION, conv4->get_oblob())->op(CACU_DROPOUT)->op(CACU_PRELU);
	conv_fc->get_op<dropout_op>(1, CACU_DROPOUT)->set_ratio(0.25);
	*net << conv_fc;
	layer *split = new layer();
	split->op(CACU_SPLIT,conv_fc->get_oblob(), new op_args(2));
	*net << split;
	layer *conv4_1 = new layer(new data_args(2, 1, 1, 0, split->get_oblobs()->at(0)->channel()));
	conv4_1->op(CACU_CONVOLUTION,  split->get_oblobs()->at(0));
	*net << conv4_1;
	layer *conv4_2 = new layer(new data_args(4, 1, 1, 0, split->get_oblobs()->at(1)->channel()));
	conv4_2->op(CACU_CONVOLUTION,  split->get_oblobs()->at(1));
	*net << conv4_2;

	layer *cls = new layer();
	if(train == phase_){
		blobs *cls_blobs = new blobs();
		cls_blobs->push_back(conv4_1->get_oblob());
		cls_blobs->push_back(label_);
		cls->op(CACU_SOFTMAX_LOSS,cls_blobs);
		cls->get_op<softmax_with_loss_op>(0, CACU_SOFTMAX_LOSS)->set_loss_weight(0.7);
	}
	else
		cls->op(CACU_SOFTMAX,conv4_1->get_oblob());
	*net << cls;

	layer *roi = new layer();
	if(train == phase_){
		blobs *roi_blobs = new blobs();
		roi_blobs->push_back(conv4_2->get_oblob());
		roi_blobs->push_back(roi_label_);
		roi->op(CACU_MSE_LOSS,roi_blobs);
		roi->get_op<mse_loss_op>(0, CACU_MSE_LOSS)->set_loss_weight(0.3);
		*net << roi;
	}


	return net;
}


#endif
