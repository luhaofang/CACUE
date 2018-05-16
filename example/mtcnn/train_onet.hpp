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

#ifndef TRAIN_ONET_HPP_
#define TRAIN_ONET_HPP_

#include <time.h>
#include <sys/time.h>
#include "../../cacu/cacu.h"

#include "../../cacu/solvers/sgd_solver.h"

#include "../../tools/string_utils.h"
#include "../../tools/imageio_utils.h"

#include "models_creater.h"

using namespace cacu_tools;
using namespace cacu;

void train_net() {
	int batch_size = 32;

	int max_iter = 100000;

	int train_test_iter = 100;

	//set gpu device if training by gpu
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(1);
#endif
#endif
	//set random seed
	set_rand_seed();

	//log output
	std::ofstream logger("/home/seal/4T/cacue/imagenet/onet.log", ios::binary);
	logger.precision(std::numeric_limits<float_t>::digits10);

	//log output
	std::ofstream precious_logger(
			"/home/seal/4T/cacue/imagenet/onet_precious.log", ios::binary);
	precious_logger.precision(std::numeric_limits<float_t>::digits10);
	//log output
	std::ofstream precious_train_logger(
			"/home/seal/4T/cacue/imagenet/onet_train_precious.log",
			ios::binary);
	precious_train_logger.precision(std::numeric_limits<float_t>::digits10);

	network *net = create_Onet(batch_size, train);

	net->load_weights("/home/seal/4T/cacue/imagenet/Onet.model");
	//net->check();
	sgd_solver *sgd = new sgd_solver(net);

	sgd->set_lr(0.0001);
	sgd->set_weight_decay(0.0001);
	sgd->set_regularize(L2);

	string datapath = "";

	vector<string> full_data;

	vector<string> vec;
	vec_i label(1);
	vec_t roi_label(4);

	/**
	 * read train list data into local memory
	 */
	string trainlist = "neg_48.txt";
	ifstream is = ifstream(trainlist);
	is.precision(numeric_limits<float>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", trainlist.c_str());
	string file_ = "";
	while (getline(is, file_)) {
		full_data.push_back((file_ + " -1 -1 -1 -1"));
	}
	is.close();
	trainlist = "pos_48.txt";
	is = ifstream(trainlist);
	is.precision(numeric_limits<float>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", trainlist.c_str());
	file_ = "";
	while (getline(is, file_)) {
		full_data.push_back(file_);
	}
	is.close();
	trainlist = "part_48.txt";
	is = ifstream(trainlist);
	is.precision(numeric_limits<float>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", trainlist.c_str());
	file_ = "";
	while (getline(is, file_)) {
		full_data.push_back(file_);
	}
	is.close();
	random_shuffle(full_data.begin(), full_data.end());

	int ALL_DATA_SIZE = full_data.size();
	/**
	 * read data for training
	 */
	blob *input_data = (blob*) net->input_blobs()->at(0);
	bin_blob *input_label = (bin_blob*) net->input_blobs()->at(1);
	blob *input_roi_label = (blob*) net->input_blobs()->at(2);
	/**
	 * read data for testing
	 */
	blob *output_data = net->output_blob();
	//blob *roi_data = net->get_op(net->op_count()-2)->out_data<blob>();

	int step_index = 0;
	int step_index_train = 0;
	struct timeval start;
	struct timeval end;
	unsigned long diff;

	unsigned int max_index;
	float_t count = 0;

	int previous_count = 0;
	int allcount = 0;

	vec_i compare_label(batch_size);
	random_shuffle(full_data.begin(), full_data.end());
	for (int i = 1; i <= max_iter; ++i) {

		if (i % train_test_iter == 0) {
			count = 0;
			step_index_train = step_index;
			gettimeofday(&start, NULL);
			for (int j = 0; j < batch_size; ++j) {

				if (step_index_train == ALL_DATA_SIZE) {
					step_index_train = 0;
				}
				file_ = full_data[step_index_train];
				vec = split(file_," ");
				label[0] = strtoul(vec[1].c_str(), NULL, 10);
				roi_label[0] = strtoul(vec[2].c_str(), NULL, 10);
				roi_label[1] = strtoul(vec[3].c_str(), NULL, 10);
				roi_label[2] = strtoul(vec[4].c_str(), NULL, 10);
				roi_label[3] = strtoul(vec[5].c_str(), NULL, 10);
				//load image data
				imageio_utils::imread_gpu(input_data->p_data(j),
						(datapath + vec[0] + ".jpg").c_str(), 3 * 48 * 48);
				compare_label[j] = label[0];
				input_label->copy2data(label, j);
				input_roi_label->copy2data(roi_label, j);
				step_index_train += 1;
			}
			cacu_sdxsize(input_data->s_data(), input_data->count(),
					(float_t) -127.5, (float_t) 128.0, input_data->s_data());
			net->predict();
			for (int j = 0; j < batch_size; ++j) {
				max_index = argmax(output_data->p_data(j),
						output_data->length());
				if (max_index == compare_label[j]) {
					count += 1.0;
				}
			}

			gettimeofday(&end, NULL);

			LOG_INFO("train_test accuracy: %f", (float_t )count / batch_size);
			precious_train_logger << count << endl;
			precious_train_logger.flush();
		}
		{
			gettimeofday(&start, NULL);
			for (int j = 0; j < batch_size; ++j) {
				if (step_index == ALL_DATA_SIZE) {
					step_index = 0;
					random_shuffle(full_data.begin(), full_data.end());
				}
				file_ = full_data[step_index];
				vec = split(file_," ");
				label[0] = strtol(vec[1].c_str(), NULL, 10);
				roi_label[0] = strtof(vec[2].c_str(), NULL);
				roi_label[1] = strtof(vec[3].c_str(), NULL);
				roi_label[2] = strtof(vec[4].c_str(), NULL);
				roi_label[3] = strtof(vec[5].c_str(), NULL);
				//load image data
				imageio_utils::imread_gpu(input_data->p_data(j),
						(datapath + vec[0] + ".jpg").c_str(), 3 * 48 * 48);
				compare_label[j] = label[0];
				input_label->copy2data(label, j);
				input_roi_label->copy2data(roi_label, j);
				step_index += 1;
			}
			cacu_sdxsize(input_data->s_data(), input_data->count(),
					(float_t) -127.5, (float_t) 128.0, input_data->s_data());
			sgd->train_iter(i);
			gettimeofday(&end, NULL);
		}

		if (i % 100 == 0) {
			diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec
					- start.tv_usec;
			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, sgd->lr(), diff / 1000);
			((mse_loss_op*) net->get_op(net->op_count() - 1))->echo();
			((softmax_with_loss_op*) net->get_op(net->op_count() - 2))->echo();
		}
		logger
				<< ((softmax_with_loss_op*) net->get_op(net->op_count() - 2))->loss()
				<< endl;
		logger.flush();

		if (i % 50000 == 0)
			sgd->set_lr_iter(0.1);

		if (i % 10000 == 0) {
			ostringstream oss;
			oss << "/home/seal/4T/cacue/imagenet/Rnet_" << i << ".model";
			net->save_weights(oss.str());
		}
	}

	logger.close();
	precious_logger.close();
	precious_train_logger.close();

	ostringstream oss;
	oss << "/home/seal/4T/cacue/imagenet/Rnet_" << max_iter << ".model";
	net->save_weights(oss.str());
	LOG_INFO("optimization is done!");

	vector<string>().swap(full_data);
	delete net;
	delete sgd;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif
