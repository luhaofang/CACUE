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

#ifndef RESNET_TEST_HPP_
#define RESNET_TEST_HPP_

#include <time.h>

#include "../../cacu/cacu.h"

#include "../../tools/imageio_utils.h"

#include "../../example/imagenet/data_proc.h"
#include "../../example/imagenet/resnet_18.h"

void weight_clip(weight *&w_){
	vec_t temp(w_->length());
	vec_t wtemp(w_->count());
	float wm = 0, ws = 0;
	cacu_sumbysize(BYWIDTH, w_->s_data(), w_->count(), (float_t)1, &wm, (float_t)0, w_->count());
	wm = wm / w_->count();
	cacu_sdxsize(w_->s_data(), w_->count(), (float_t)-1 * wm, (float_t)1, &wtemp[0]);
	cacu_sqr(&wtemp[0], w_->count(), &wtemp[0]);
	cacu_sumbysize(BYWIDTH, &wtemp[0], w_->count(), (float_t)1, &ws, (float_t)0, w_->count());
	ws = sqrtf(ws / w_->count());

	for(int i = 0; i < w_->num(); ++i){
		float m = 0, s = 0;
		cacu_sumbysize(BYWIDTH, w_->p_data(i), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
		m = m / w_->length();
		cacu_sdxsize(w_->p_data(i), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
		cacu_sqr(&temp[0], w_->length(), &temp[0]);
		cacu_sumbysize(BYWIDTH, &temp[0], w_->length(), (float_t)1, &s, (float_t)0, w_->length());
		s = sqrtf(s / w_->length());
		LOG_DEBUG("[%.8f, %.8f, %.8f, %.8f]", m, s, wm, ws);
		if(s < ws / w_->length()){
			w_->set_pdata(i, 0);
			LOG_DEBUG("delete: %d", i);
		}
	}
	w_->blob_size();
	//cacu_print(w_->s_data(), w_->count());

}

void test_net()
{
	int batch_size = 1;

	int ALLIMAGE = 10000;

	int max_iter = 10000;

#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif
	blobs *inputs = new blobs();
	inputs->push_back(new blob(batch_size, 3, 28, 28, 0, test));
	network *net = create_res20net_fat(batch_size, 2, test);//phrase_network("/Users/seallhf/Documents/datasets/cifar10/res18net_cifar.ch", inputs);
//	network *net = phrase_network("/Users/seallhf/Documents/datasets/cifar10/res20_pruned.ch", inputs);
//	net->set_is_use_bias(false);
	net->load_weights("/home/seallhf/4T/dataset/cifar/resnet_100_20.model");
//	net->load_weights("/Users/seallhf/Documents/datasets/cifar10/res20_pruned.model");

	//net->weights_pre_proc(weight_clip);

	string datapath = "/home/seallhf/4T/dataset/cifar/32x32_bmp/val/";
	string vallist = "/home/seallhf/4T/dataset/cifar/test_list.txt";
	string meanfile = "/home/seallhf/4T/dataset/cifar/mean.binproto";

	vector<string> full_data;
	vector<unsigned int> full_label;

	/**
	 * load mean data
	 */
	blob *mean_ = cacu_allocator::create_blob(1,3,32,32,test);
	#if __PARALLELTYPE__ == __CUDA__
		imageio_utils::load_mean_file_gpu(mean_->s_data(),meanfile);
	#else
		imageio_utils::load_mean_file(mean_->s_data(),meanfile);
	#endif


	/**
	 * read test list data into local memory
	*/
	ifstream is(vallist);
	is.precision(numeric_limits<float>::digits10);
	if(!is)
		LOG_FATAL("file %s cannot be opened!",vallist.c_str());
	string file_;
	while(getline(is,file_)){
		vector<string> vec = split(file_," ");
		full_data.push_back(datapath + vec[0]);
		unsigned int label = strtoul(vec[1].c_str(), NULL, 10);
		full_label.push_back(label);
	}
	is.close();


	/**
	 * read data for testing
	 */
	blob *input_data = (blob*)net->input_blobs()->at(0);
	blob *output_data = net->output_blob();

	unsigned int max_index;
	cacu::float_t count = 0;

	int step_index = 0;

	time_utils *timer = new time_utils();

	vec_t channels(3);
	channels[0] = -1;
	channels[1] = -1;
	channels[2] = -1;

	vector<relu_op*> relu_ops = net->get_ops<relu_op>();

	ostringstream oss;

	unsigned long diff;
	for (int i = 0 ; i < max_iter; ++i)
	{
		oss.str("");
		oss << "/home/seallhf/4T/experiments/layer_feature_cifar/" << i << ".txt";

		std::ofstream logger(oss.str(), ios::binary);
		logger.precision(std::numeric_limits<cacu::float_t>::digits10);

		for (int j = 0 ; j < batch_size ; ++j)
		{
			if (step_index == ALLIMAGE)
				break;
			//load image data
			//readdata(full_data[step_index].c_str(),input_data->p_data(j),NULL);
			imageio_utils::center_clip_imread_channel_wise_norm(input_data->p_data(j),full_data[step_index].c_str(), 28, 28, channels);


			//readdata("/home/seal/4T/imagenet/test1.JPEG",input_data->p_data(j));//,mean_->s_data());
			step_index += 1;
		}

		timer->start();

		net->predict();

		timer->end();

		for (int j = 0; j < relu_ops.size(); ++j){
			for (int c = 0; c < relu_ops[j]->out_datas()->asblob(0)->length(); ++c){
				logger << relu_ops[j]->out_datas()->asblob(0)->s_data()[c] << ",";
			}
			logger << endl;
		}

		if (step_index == ALLIMAGE)
			break;
		logger.close();
	}

	LOG_INFO("precision: %f,%f", count / ALLIMAGE, count);

	//injector->o_blob_serializa("/home/seal/4T/cacue/imagenet/relu.txt");

	//delete injector;
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
}


#endif
