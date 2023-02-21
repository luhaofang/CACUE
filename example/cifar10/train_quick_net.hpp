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

#ifndef TRAIN_QUICK_NET_HPP_
#define TRAIN_QUICK_NET_HPP_

#include <time.h>
#include <sys/time.h>
#include "../../cacu/cacu.h"

#include "../../cacu/solvers/sgd_solver.h"

#include "../../tools/string_utils.h"
#include "../../tools/imageio_utils.h"

//#include "vgg_net.h"
#include "../imagenet/data_proc.h"
#include "cifar_quick_net.h"
//#include "cifar_test_net.h"

using namespace cacu;

void weight_diff_recode(weight *&w_, std::ostream &os){
	if(w_->length() == 1)
		return;
	vec_t temp(w_->length());
	vec_t wtemp(w_->count());
	w_->update_index()->clear();
	int count = 0;
	float_t acc_cdf = 0;
	float_t acc_md = 0;
	float_t acc_sd = 0;

	for(int i = 0; i < w_->num(); ++i){
		float_t m = 0, s = 0;
		float_t md = 0, sd = 0;
#if __USE_DEVICE__ == ON
		cacu_sumbysize_cpu(BYWIDTH, w_->p_data_cpu(i, false), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
		cacu_sumbysize_cpu(BYWIDTH, w_->p_diff_cpu(i), w_->length(), (float_t)1, &md, (float_t)0, w_->length());
#else
		cacu_sumbysize(BYWIDTH, w_->p_data(i), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
		cacu_sumbysize(BYWIDTH, w_->p_diff(i), w_->length(), (float_t)1, &md, (float_t)0, w_->length());
#endif
		m = m / w_->length();
		md = md / w_->length();
#if __USE_DEVICE__ == ON
		cacu_sdxsize_cpu(w_->p_data_cpu(i, false), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
#else
		cacu_sdxsize(w_->p_data(i), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
#endif
		cacu_sqr_cpu(&temp[0], w_->length(), &temp[0]);
		cacu_sumbysize_cpu(BYWIDTH, &temp[0], w_->length(), (float_t)1, &s, (float_t)0, w_->length());
		s = sqrtf(s / (w_->length() - 1));

#if __USE_DEVICE__ == ON
		cacu_sdxsize_cpu(w_->p_diff_cpu(i, false), w_->length(), (float_t)-1 * md, (float_t)1, &temp[0]);
#else
		cacu_sdxsize(w_->p_diff(i), w_->length(), (float_t)-1 * md, (float_t)1, &temp[0]);
#endif
		cacu_sqr_cpu(&temp[0], w_->length(), &temp[0]);
		cacu_sumbysize_cpu(BYWIDTH, &temp[0], w_->length(), (float_t)1, &sd, (float_t)0, w_->length());
		sd = sqrtf(sd / (w_->length() - 1));

		acc_sd += sd / w_->num();
		acc_md += md / w_->num();
		os << md << '|' << sd << '#';
	}
	os << acc_md<< '|' << acc_sd << ",";
}


void weight_recode(weight *&w_, std::ostream &os){
	if(w_->length() == 1)
		return;
	vec_t temp(w_->length());
	vec_t wtemp(w_->count());
	w_->update_index()->clear();
	int count = 0;
	float_t acc_cdf = 0;
	float_t acc_md = 0;
	float_t acc_sd = 0;

	for(int i = 0; i < w_->num(); ++i){
		float_t m = 0, s = 0;
		float_t md = 0, sd = 0;
#if __USE_DEVICE__ == ON
		cacu_sumbysize_cpu(BYWIDTH, w_->p_data_cpu(i, false), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
		cacu_sumbysize_cpu(BYWIDTH, w_->p_diff_cpu(i), w_->length(), (float_t)1, &md, (float_t)0, w_->length());
#else
		cacu_sumbysize(BYWIDTH, w_->p_data(i), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
		cacu_sumbysize(BYWIDTH, w_->p_diff(i), w_->length(), (float_t)1, &md, (float_t)0, w_->length());
#endif
		m = m / w_->length();
		md = md / w_->length();
#if __USE_DEVICE__ == ON
		cacu_sdxsize_cpu(w_->p_data_cpu(i, false), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
#else
		cacu_sdxsize(w_->p_data(i), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
#endif
		cacu_sqr_cpu(&temp[0], w_->length(), &temp[0]);
		cacu_sumbysize_cpu(BYWIDTH, &temp[0], w_->length(), (float_t)1, &s, (float_t)0, w_->length());
		s = sqrtf(s / (w_->length() - 1));

#if __USE_DEVICE__ == ON
		cacu_sdxsize_cpu(w_->p_diff_cpu(i, false), w_->length(), (float_t)-1 * md, (float_t)1, &temp[0]);
#else
		cacu_sdxsize(w_->p_diff(i), w_->length(), (float_t)-1 * md, (float_t)1, &temp[0]);
#endif
		cacu_sqr_cpu(&temp[0], w_->length(), &temp[0]);
		cacu_sumbysize_cpu(BYWIDTH, &temp[0], w_->length(), (float_t)1, &sd, (float_t)0, w_->length());
		sd = sqrtf(sd / (w_->length() - 1));

		acc_sd += s / w_->num();
		acc_md += m / w_->num();
		os << m << '|' << s << '#';
	}
	os << acc_md<< '|' << acc_sd << ",";
}

void weight_rotate(weight *&w_){
	w_->switch_random_rotate(0);
}

void train_net() {
	int batch_size = 100;

	int max_iter = 30000;

	int test_iter = 500;
	int train_test_iter = 500;

	//set gpu device if training by gpu
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif
#endif
	//set random seed
	set_rand_seed();

	//log output
	std::ofstream logger("/Users/seallhf/Documents/datasets/cifar10/resnet_20_x7.log",
			ios::binary);
	logger.precision(std::numeric_limits<float_t>::digits10);

	//log output
	std::ofstream precious_logger(
			"/Users/seallhf/Documents/datasets/cifar10/resnet_precious_20_x7.log",
			ios::binary);
	precious_logger.precision(std::numeric_limits<float_t>::digits10);
	//log output
	std::ofstream precious_train_logger(
			"/Users/seallhf/Documents/datasets/cifar10/resnet_train_precious_20_x7.log",
			ios::binary);
	precious_train_logger.precision(std::numeric_limits<float_t>::digits10);

	//log output
	std::ofstream gradient_logger(
			"/Users/seallhf/Documents/datasets/cifar10/gradient_rand_x5.log",
			ios::binary);
	gradient_logger.precision(std::numeric_limits<float_t>::digits10);

	//log output
	std::ofstream filter_logger(
			"/Users/seallhf/Documents/datasets/cifar10/filters_rand_x7.log",
			ios::binary);
	filter_logger.precision(std::numeric_limits<float_t>::digits10);

	//network *net = create_mobilenet(batch_size,train);
	//network *net = create_vgg_16_net(batch_size, train);
	network *net = create_cifar_3convs_net(batch_size,train);//create_mobilenet(batch_size,train);//create_alexnet(batch_size,train);

//	net->load_weights("/Users/seallhf/Documents/datasets/cifar10/quick_net_20000.model");
	//net->check();
	sgd_solver *sgd = new sgd_solver(net);

	sgd->set_lr(0.001);
	sgd->set_weight_decay(0.002);
	sgd->set_regularize(cacu::regularize_type::L2);
//	sgd->set_positive_regularize(true);
//	sgd->set_train_stage(PRUNE_STAGE);

	string datapath = "/Users/seallhf/Documents/datasets/cifar10/32x32/train/";
	string trainlist = "/Users/seallhf/Documents/datasets/cifar10/train_list.txt";
	string valdatapath = "/Users/seallhf/Documents/datasets/cifar10/32x32/val/";
	string vallist = "/Users/seallhf/Documents/datasets/cifar10/test_list.txt";
	string meanfile = "/Users/seallhf/Documents/datasets/cifar10/mean.binproto";

	vector<string> full_data;

	vector<string> full_data_test;
	vector<vec_i> full_label_test;

	vector<string> vec;
	vec_i label(1);
	/**
	 * load mean data
	 */
	blob *mean_ = cacu_allocator::create_blob(1, 3, 32, 32, test);
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	imageio_utils::load_mean_file_gpu(mean_->s_data(), meanfile);
	float_t *mean_data_ = mean_->s_data_cpu();
#endif
#else
	imageio_utils::load_mean_file(mean_->s_data(),meanfile);
	float_t *mean_data_ = mean_->s_data();
#endif
	/**
	 * read train list data into local memory
	 */
	ifstream is = ifstream(trainlist);
	is.precision(numeric_limits<float>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", trainlist.c_str());
	string file_ = "";
	while (getline(is, file_)) {
		full_data.push_back(file_);
	}
	is.close();
//	random_shuffle(full_data.begin(), full_data.end());

	/**
	 * read test list data into local memory
	 */
	is = ifstream(vallist);
	is.precision(numeric_limits<float>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", vallist.c_str());
	file_ = "";
	while (getline(is, file_)) {
		vec = split(file_, " ");
		full_data_test.push_back(valdatapath + vec[0]);
		label[0] = strtoul(vec[1].c_str(), NULL, 10);
		full_label_test.push_back(label);
	}
	is.close();
	int ALL_DATA_SIZE = full_data.size();
	int ALL_TEST_DATA_SIZE = full_data_test.size();
	/**
	 * read data for training
	 */
	blob *input_data = (blob*) net->input_blobs()->at(0);
	bin_blob *input_label = (bin_blob*) net->input_blobs()->at(1);

	/**
	 * read data for testing
	 */
	blob *output_data = net->output_blob();

	int step_index = 0;
	int step_index_test = 0;
	int step_index_train = 0;
	struct timeval start;
	struct timeval end;
	unsigned long diff;

	unsigned int max_index;
	float_t count = 0;

	int previous_count=0;
	int allcount = 0;

	vec_t channels(3);
//	channels[0] = 122.7717;
//	channels[1] = 115.9465;
//	channels[2] = 102.9801;
	channels[0] = -1;
	channels[1] = -1;
	channels[2] = -1;


	vec_i compare_label(batch_size);
	random_shuffle(full_data.begin(), full_data.end());
	for (int i = 1; i <= max_iter; ++i) {

		net->weights_pre_proc(weight_rotate);

		if (i % test_iter == 0) {
			count = 0;
			gettimeofday(&start, NULL);
			for (int test_c = 0; test_c < 100; test_c++) {
				for (int j = 0; j < batch_size; ++j) {
					if (step_index_test == ALL_TEST_DATA_SIZE)
						step_index_test = 0;
					//load image data
					//readdata(full_data_test[step_index_test].c_str(),input_data->p_data(j), mean_->s_data());
					//(full_data_test[step_index_test].c_str(),input_data->p_data(j), NULL);
//					imageio_utils::center_clip_imread_channel_wise_norm(input_data->p_data(j),full_data_test[step_index_test].c_str(), 32, 32, channels);
					imageio_utils::center_clip_imread_element_wise_norm(input_data->p_data(j),full_data_test[step_index_test].c_str(), 32, 32, mean_data_);
					input_label->copy2data(full_label_test[step_index_test],
							j);
					compare_label[j] = full_label_test[step_index_test][0];
					step_index_test += 1;
				}
				net->predict();
				for (int j = 0; j < batch_size; ++j) {
					max_index = argmax(output_data->p_data(j),
							output_data->length());
					if (max_index == compare_label[j]) {
						count += 1.0;
					}
				}
			}
			gettimeofday(&end, NULL);

			//if(previous_count >= count)
			//	sgd->set_lr_iter(0.1f);

			previous_count = count;

			LOG_INFO("test accuracy: %f", (float_t)count/batch_size/100);
			precious_logger << count << endl;
			precious_logger.flush();
		}
		if (i % train_test_iter == 0)
		{
			count = 0;
			step_index_train = step_index;
			gettimeofday(&start, NULL);
			for (int j = 0; j < batch_size; ++j) {

				if (step_index_train == ALL_DATA_SIZE) {
					step_index_train = 0;
//					random_shuffle(full_data.begin(), full_data.end());
				}
				file_ = full_data[step_index_train];
				vec = split(file_, " ");
				label[0] = strtoul(vec[1].c_str(), NULL, 10);
				//load image data
				//clipreaddata((datapath + vec[0]).c_str(), input_data->p_data(j), mean_->s_data());
				//clipreaddata((datapath + vec[0]).c_str(), input_data->p_data(j), NULL);
//				imageio_utils::clip_imread_channel_wise_norm(input_data->p_data(j),(datapath + vec[0]).c_str(), 32, 32, channels, false);
				imageio_utils::clip_imread_element_wise_norm(input_data->p_data(j),(datapath + vec[0]).c_str(), 32, 32, mean_data_, false);

				compare_label[j] = label[0];
				input_label->copy2data(label, j);
				step_index_train += 1;
			}
			net->predict();
			for (int j = 0; j < batch_size; ++j) {
				max_index = argmax(output_data->p_data(j),
						output_data->length());
				if (max_index == compare_label[j]) {
					count += 1.0;
				}
			}
			//sgd->train_iter();

			gettimeofday(&end, NULL);

			//if(previous_count >= count)
			//	sgd->set_lr_iter(0.1f);

			LOG_INFO("train_test accuracy: %f", (float_t)count/batch_size);
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
				vec = split(file_, " ");
				label[0] = strtoul(vec[1].c_str(), NULL, 10);
				//load image data
				//clipreaddata((datapath + vec[0]).c_str(), input_data->p_data(j), mean_->s_data());
				//clipreaddata((datapath + vec[0]).c_str(), input_data->p_data(j), NULL);
//				imageio_utils::clip_imread_channel_wise_norm(input_data->p_data(j),(datapath + vec[0]).c_str(), 32, 32, channels, false);
				imageio_utils::clip_imread_element_wise_norm(input_data->p_data(j),(datapath + vec[0]).c_str(), 32, 32, mean_data_, false);

				input_label->copy2data(label, j);
				step_index += 1;
			}
//			net->predict();
//			sgd->train_iter(i);
			net->set_phase(train);

			net->forward_propagate();

			net->backward_propagate();

			for(int op_i = 0 ; op_i < net->op_count(); ++op_i){
				sgd->data_recode(weight_recode, filter_logger, op_i);
				sgd->data_recode(weight_diff_recode, gradient_logger, op_i);
			}

			sgd->updates(i);
			gradient_logger << endl;
			filter_logger << endl;
			gettimeofday(&end, NULL);
		}

		if (i % 1 == 0) {
			diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec
					- start.tv_usec;
			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, sgd->lr(), diff / 1000);
			((softmax_with_loss_op*) net->get_op(net->op_count() - 1))->echo();
		}
		logger
				<< ((softmax_with_loss_op*) net->get_op(net->op_count() - 1))->loss()
				<< endl;
		logger.flush();

		if (i % 24000 == 0)
			sgd->set_lr_iter(0.1);

		if (i % 10000 == 0) {
			ostringstream oss;
			oss << "/Users/seallhf/Documents/datasets/cifar10/quick_net_" << i << ".model";
			net->save_weights(oss.str());
		}
	}

	ostringstream oss;
	oss << "/Users/seallhf/Documents/datasets/cifar10/quick_net.model";
	net->save_weights(oss.str());

	logger.close();
	precious_logger.close();
	precious_train_logger.close();
	gradient_logger.close();
	filter_logger.close();

	LOG_INFO("optimization is done!");

	for (int i = 0; i < full_label_test.size(); ++i) {
		vec_i().swap(full_label_test[i]);
	}
	vector<string>().swap(full_data);
	vector<string>().swap(full_data_test);
	delete mean_;
	delete net;
	delete sgd;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif
