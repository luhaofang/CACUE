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


#ifndef OUTPUT_FEATURE_HPP_
#define OUTPUT_FEATURE_HPP_

#include "../../cacu/cacu.h"

#include "../../tools/imageio_utils.h"
#include "../../tools/time_utils.h"

#include "cifar_quick_net.h"
#include "data_proc.h"

using namespace cacu;
using namespace cacu_tools;

void weight_clip(weight *&w_){
	if(w_->length() == 1)
		return;
	vec_t temp(w_->length());
	vec_t wtemp(w_->count());
	float wm = 0, ws = 0;
	cacu_sumbysize(BYWIDTH, w_->s_data(), w_->count(), (float_t)1, &wm, (float_t)0, w_->count());
	wm = wm / w_->count();
	cacu_sdxsize(w_->s_data(), w_->count(), (float_t)-1 * wm, (float_t)1, &wtemp[0]);
	cacu_sqr(&wtemp[0], w_->count(), &wtemp[0]);
	cacu_sumbysize(BYWIDTH, &wtemp[0], w_->count(), (float_t)1, &ws, (float_t)0, w_->count());
	ws = sqrtf(ws / (w_->count()));

	int count = 0;

	for(int i = 0; i < w_->num(); ++i){
		float m = 0, s = 0;
		cacu_sumbysize(BYWIDTH, w_->p_data(i), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
		m = m / w_->length();
		cacu_sdxsize(w_->p_data(i), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
		cacu_sqr(&temp[0], w_->length(), &temp[0]);
		cacu_sumbysize(BYWIDTH, &temp[0], w_->length(), (float_t)1, &s, (float_t)0, w_->length());
		s = sqrtf(s / (w_->length()));
		LOG_DEBUG("[%.8f, %.8f, %.8f, %.8f]", m, s, wm, ws);
		if(s < ws / w_->length()){
			count += 1;
//			cacu_set_value(w_->p_data(i), 0, w_->length());
//			LOG_DEBUG("delete: %d", i);
		}
	}
	LOG_DEBUG("%d", count);
	w_->blob_size();
	//cacu_print(w_->s_data(), w_->count());

}

void test_net()
{
	int batch_size = 100;

	int max_iter = 500;

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif
#endif

	string datapath = "/Users/seallhf/Documents/datasets/cifar10/";
	string rootpath = "/Users/seallhf/Documents/datasets/cifar10/";

//	network *net = create_cifar_3convs_net(batch_size,test);
	network *net = create_cifar_quick_net(batch_size,test);
//	net->set_is_use_bias(true);
	net->load_weights(datapath + "cifar10_quick_test.model");

	std::ofstream logger(rootpath + "feature_regularized.txt", ios::binary);
		logger.precision(std::numeric_limits<cacu::float_t>::digits10);

//	blobs *inputs = new blobs();
//	inputs->push_back(new blob(1, 3, 32, 32, 0, test));
//	network *net = phrase_network("/Users/seallhf/Documents/datasets/cifar10/cifar10_quick_test_pruned.ch", inputs);
//	net->set_is_use_bias(false);
////	net->load_weights("/Users/seallhf/Documents/datasets/cifar10/cifar10_quick_test_pruned_wg.model");
//	net->load_weights(datapath + "cifar10_quick_test_pruned.model");

	string meanfile = datapath + "mean.binproto";

	vector<vec_t> full_data;
	vector<vec_i> full_label;
	load_data_bymean(datapath, meanfile, full_data, full_label);

	vec_i _full_label;
	for(int i = 0; i < full_label.size(); ++i)
		_full_label.push_back(full_label[i][0]);

	blob *input_data = (blob*)net->input_blobs()->at(0);

	blob *output_data = net->output_blob();

//	net->weights_pre_proc(weight_clip);
//	cacu_print(net->get_op<convolution_op>(0)->get_weight(0)->s_data(), 3*32*5*5);

//	vector<convolution_op*> convs = net->get_ops<convolution_op>();
//	vector<inner_product_op*> ips = net->get_ops<inner_product_op>();
//
//	for(int i = 0; i < convs.size(); ++i)
//		convs[i]->set_is_norm_weights(false, 1);
//
//	for(int i = 0; i < ips.size(); ++i)
//		ips[i]->set_is_norm_weights(false, 1);

	unsigned int max_index;
	cacu::float_t count = 0;
	time_utils *timer= new time_utils();
	unsigned long diff;

	int step_index = 0;

	int allcount = 0;
	for (int i = 0; i < max_iter; ++i)
	{
		timer->start();

		input_data->resize(batch_size, 3, 32, 32);
		for (int j = 0; j < batch_size; ++j)
		{
			if (step_index == kCIFARBatchSize)
				break;
			input_data->copy2data(full_data[step_index], j);
			step_index += 1;
		}

		net->predict();
		for (int j = 0; j < batch_size; ++j)
		{
			float_t *data_ = net->get_op<inner_product_op>(net->op_count() - 3)->out_data<blob>()->p_data(j);
			logger << _full_label[allcount + j] << " ";
			for(int m = 0; m < 64 ; ++m)
				logger << data_[m] << ",";
			logger << endl;
		}
		allcount += batch_size;
//		batch_size = 100;//urandint(10, 100);
//		LOG_DEBUG("batch_size: %d", batch_size);
		timer->end();

		if (i % 1 == 0) {
			LOG_INFO("iter_%d, %ld ms/iter", i, timer->get_time_span() / 1000);
		}
//		if (step_index == kCIFARBatchSize)
//			break;
	}
//	imageio_utils::imwrite(input_data, (datapath + "test.jpg").c_str());
//	imageio_utils::imwrite_channel(net->get_op(0)->out_data<blob>(), (datapath + "cifar10_quick_test.jpg").c_str());

//	LOG_INFO("precision: %f,%f", count / allcount,count);
	logger.close();
	delete net;
	delete timer;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif
