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


#ifndef TRAIN_STD_HPP_
#define TRAIN_STD_HPP_

#include <time.h>

#include "../../cacu/solvers/sgd_solver.h"
#include "../../cacu/solvers/adam_solver.h"

#include "../../cacu/cacu.h"
#include "../../cacu/config.h"

#include "../../tools/imageio_utils.h"
#include "../../tools/time_utils.h"

#include "cifar_quick_net.h"
//#include "cifar_test_net.h"
#include "data_proc.h"
#include "../../tools/vec_utils.h"

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

	network *net = create_cifar_quick_net(batch_size,train);

	float_t alpha_decay = 0.85;
	sgd_solver *sgd = new sgd_solver(net);
	sgd->set_regularize(cacu::regularize_type::L2);

	sgd->set_lr(0.001);
	sgd->set_weight_decay(0.004);

	string rootpath = "/Users/seallhf/Documents/datasets/cifar10/";
//	net->load_weights(rootpath + "cifar10_quick_test.model");

	std::ofstream diff_log(rootpath + "diff_log.txt", ios::binary);
	diff_log.precision(std::numeric_limits<cacu::float_t>::digits10);

	std::ofstream logger(rootpath + "loss_positive.txt", ios::binary);
	logger.precision(std::numeric_limits<cacu::float_t>::digits10);

	std::ofstream clogger(rootpath + "loss_center.txt", ios::binary);
	logger.precision(std::numeric_limits<cacu::float_t>::digits10);

	std::ofstream p_logger(rootpath + "train_precision_positive.txt", ios::binary);
	p_logger.precision(std::numeric_limits<cacu::float_t>::digits10);

	string meanfile = rootpath + "mean.binproto";

	vector<vec_t> full_data;
	vector<vec_i> full_label;

	load_data_bymean(rootpath, meanfile, full_data, full_label);

	vector<vec_t> full_data_test;
	vector<vec_i> full_label_test;
	load_test_data_bymean(rootpath, meanfile, full_data_test, full_label_test);

	vec_i _full_label;
	for(int i = 0; i < full_label.size(); ++i)
		_full_label.push_back(full_label[i][0]);

	vec_i _full_label_test;
	for(int i = 0; i < full_label_test.size(); ++i)
		_full_label_test.push_back(full_label_test[i][0]);
	//load_data(datapath, full_data, full_label);

	blob *input_data = (blob*)net->input_blobs()->at(0);
	bin_blob *input_label = (bin_blob*)net->input_blobs()->at(1);

	blob *output_data = net->output_blob();

	int step_index = 0;
	int step_index_test = 0;
	unsigned int max_index;
	time_utils *timer = new time_utils();
	unsigned long diff;

	int allcount = 0;
	float_t count = 0;

	for (int i = 1 ; i < max_iter; ++i)
	{
		timer->start();

		for (int j = 0 ; j < batch_size ; ++j)
		{
			if (step_index == kCIFARDataCount)
				step_index = 0;
			input_data->copy2data(full_data[step_index], j);
			input_label->copy2data(full_label[step_index],j);
			step_index += 1;
		}

//		sgd->train_iter(i);
		sgd->net()->set_phase(train);

		sgd->net()->forward_propagate();

		sgd->net()->backward_propagate();


		diff_log << endl;

		sgd->updates(i);

		timer->end();

		if(i % 1 == 0){

			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, sgd->lr(), timer->get_time_span() / 1000);
			((softmax_with_loss_op*)net->get_op(net->op_count() - 1))->echo();
//			net->get_op<center_loss_op>(net->op_count() - 3)->echo();

			logger << ((softmax_with_loss_op*)net->get_op(net->op_count() - 1))->loss() << endl;
			logger.flush();

//			clogger << net->get_op<center_loss_op>(net->op_count() - 3)->loss() << endl;
//			clogger.flush();
		}

		if(i % 500 == 0){
			step_index_test = 0;
			allcount = 0;
			count = 0;
			for (int n = 0; n < 100; ++n){
				timer->start();
				//input_data->resize(batch_size, 3, 32, 32);
				for (int j = 0; j < batch_size; ++j){
					if (step_index_test == kCIFARBatchSize)
						break;
					input_data->copy2data(full_data_test[step_index_test], j);
					step_index_test += 1;
				}

				net->predict();
				for (int j = 0; j < batch_size; ++j){
					max_index = argmax(output_data->p_data(j), output_data->length());
					if (max_index == _full_label_test[allcount + j]) {
						count += 1.0;
					}
				}
				allcount += batch_size;
				timer->end();

				if (step_index_test == kCIFARBatchSize)
					break;
			}
			p_logger << count << endl;
			p_logger.flush();
		}

		if(i % 4000 == 0){
			sgd->set_lr_iter(0.1);
		}
	}
	LOG_INFO("optimization is done!");
	net->save_weights(rootpath + "cifar10_quick_std.model");

	vector<vec_t>().swap(full_data);
	vector<vec_i>().swap(full_label);
	logger.close();
	clogger.close();
	p_logger.close();
	diff_log.close();
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

