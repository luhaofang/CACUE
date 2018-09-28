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


#ifndef TRAIN_HPP_
#define TRAIN_HPP_

#include <time.h>

#include "../../cacu/solvers/sgd_solver.h"
#include "../../cacu/solvers/adam_solver.h"

#include "../../cacu/cacu.h"
#include "../../cacu/config.h"

#include "../../tools/imageio_utils.h"
#include "../../tools/time_utils.h"

#include "lenet.h"
#include "mnist_data_proc.h"

using namespace cacu;
using namespace cacu_tools;

void train_net()
{
	int batch_size = 100;

	int max_iter = 5000;

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);

#endif
#endif
	//set random seed
	set_rand_seed();

	network *net = create_lenet(batch_size,train);

	sgd_solver *sgd = new sgd_solver(net);
	sgd->set_lr(0.01f);
	sgd->set_momentum(0.9f);
	sgd->set_weight_decay(0.0005f);

	string datapath = "/home/luhaofang/git/caffe/data/mnist/";

	std::ofstream logger(datapath + "loss.txt", ios::binary);
	logger.precision(std::numeric_limits<cacu::float_t>::digits10);

	string meanfile = datapath + "mean.binproto";

	vector<vec_t> full_data;
	vector<vec_i> full_label;

	load_data_bymean_mnist(datapath, meanfile, full_data, full_label);
	//load_data(datapath, full_data, full_label);

	blob *input_data = (blob*)net->input_blobs()->at(0);
	bin_blob *input_label = (bin_blob*)net->input_blobs()->at(1);

	int step_index = 0;
	time_utils *timer = new time_utils();
	unsigned long diff;
	for (int i = 1 ; i < max_iter; ++i)
	{
		timer->start();

		for (int j = 0 ; j < batch_size ; ++j)
		{
			if (step_index == kMNISTDataCount)
				step_index = 0;
			input_data->copy2data(full_data[step_index], j);
			input_label->copy2data(full_label[step_index],j);
			step_index += 1;
		}
		
		sgd->train_iter(i);
		//cacu_print(net->get_op<inner_product_op>(net->op_count() - 2, CACU_INNERPRODUCT)->out_data<blob>()->s_data(), 10);

		timer->end();

		if(i % 10 == 0){

			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, sgd->lr(), timer->get_time_span() / 1000);
			net->get_op<softmax_with_loss_op>(net->op_count() - 1, CACU_SOFTMAX_LOSS)->echo();

			logger << net->get_op<softmax_with_loss_op>(net->op_count() - 1, CACU_SOFTMAX_LOSS)->loss() << endl;
			logger.flush();
		}

		if(i % 4000 == 0)
			sgd->set_lr_iter(0.1f);

	}
	LOG_INFO("optimization is done!");
	net->save_weights(datapath + "lenet.model");

	vector<vec_t>().swap(full_data);
	vector<vec_i>().swap(full_label);
	logger.close();
	delete net;
	delete sgd;

	delete timer;

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif

