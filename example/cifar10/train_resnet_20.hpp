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

#ifndef TRAIN_RESNET_20_HPP_
#define TRAIN_RESNET_20_HPP_

#include <time.h>
#include <sys/time.h>
#include "../../cacu/cacu.h"
//#include "../../cacu/tensor/utils/device_data_utils.h"

#include "../../cacu/solvers/sgd_solver.h"

#include "../../tools/string_utils.h"
#include "../../tools/imageio_utils.h"
#include "../../cacu/framework/initialize_functions.h"

//#include "vgg_net.h"
#include "../imagenet/data_proc.h"
#include "../imagenet/resnet_18.h"
#include "../imagenet/resnet_50.h"
#include "../imagenet/mobilenet.h"

using namespace cacu;

//upgrade training for initial convolution & fc weights
void weight_initial(weight *&w_) {
	if (w_->length() == 1)
		return;
	vec_t temp(w_->length());
	vec_t wtemp(w_->count());
	int count = 0, s_count = 0;
	float wm = 0, ws = 0;
#if __USE_DEVICE__ == ON
	cacu_sumbysize_cpu(BYWIDTH, w_->s_data_cpu(), w_->count(), (float_t) 1, &wm,
			(float_t) 0, w_->count());
#else
	cacu_sumbysize(BYWIDTH, w_->s_data(), w_->count(), (float_t)1, &wm, (float_t)0, w_->count());
#endif
	wm = wm / w_->count();
#if __USE_DEVICE__ == ON
	cacu_sdxsize_cpu(w_->s_data_cpu(false), w_->count(), (float_t) -1 * wm,
			(float_t) 1, &wtemp[0]);
	cacu_sqr_cpu(w_->s_data_cpu(false), w_->count(), &wtemp[0]);
#else
	cacu_sdxsize(w_->s_data(), w_->count(), (float_t)-1 * wm, (float_t)1, &wtemp[0]);
	cacu_sqr(w_->s_data(), w_->count(), &wtemp[0]);
#endif
	cacu_sumbysize_cpu(BYWIDTH, &wtemp[0], w_->count(), (float_t) 1, &ws,
			(float_t) 0, w_->count());
	ws = sqrtf(ws / (w_->count() - 1));
	//w_->set_init_type(guassian, ws);

	for (int i = 0; i < w_->num(); ++i) {
		float m = 0, s = 0;
#if __USE_DEVICE__ == ON
		cacu_sumbysize_cpu(BYWIDTH, w_->p_data_cpu(i, false), w_->length(),
				(float_t) 1, &m, (float_t) 0, w_->length());
#else
		cacu_sumbysize(BYWIDTH, w_->p_data(i), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
#endif
		m = m / w_->length();
#if __USE_DEVICE__ == ON
		cacu_sdxsize_cpu(w_->p_data_cpu(i, false), w_->length(),
				(float_t) -1 * m, (float_t) 1, &temp[0]);
		cacu_sqr_cpu(w_->p_data_cpu(i, false), w_->length(), &temp[0]);
#else
		cacu_sdxsize(w_->p_data(i), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
		cacu_sqr(w_->p_data(i), w_->length(), &temp[0]);
#endif
		cacu_sumbysize_cpu(BYWIDTH, &temp[0], w_->length(), (float_t) 1, &s,
				(float_t) 0, w_->length());
		s = sqrtf(s / (w_->length() - 1));
		//LOG_DEBUG("[%.8f, %.8f, %.8f, %.8f]", m, s, wm, ws);

		if (s < ws / w_->length()) {
			gaussian_init(w_->p_data(i), w_->length(), wm, ws * 0.7);
			//LOG_DEBUG("copy: %d", i);
		}
	}

	for (int i = 0; i < w_->num(); ++i) {
		float m = 0, s = 0;
#if __USE_DEVICE__ == ON
		cacu_sumbysize_cpu(BYWIDTH, w_->p_data_cpu(i), w_->length(),
				(float_t) 1, &m, (float_t) 0, w_->length());
#else
		cacu_sumbysize(BYWIDTH, w_->p_data(i), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
#endif
		m = m / w_->length();
#if __USE_DEVICE__ == ON
		cacu_sdxsize_cpu(w_->p_data_cpu(i, false), w_->length(),
				(float_t) -1 * m, (float_t) 1, &temp[0]);
		cacu_sqr_cpu(w_->p_data_cpu(i, false), w_->length(), &temp[0]);
#else
		cacu_sdxsize(w_->p_data(i), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
		cacu_sqr(w_->p_data(i), w_->length(), &temp[0]);
#endif
		cacu_sumbysize_cpu(BYWIDTH, &temp[0], w_->length(), (float_t) 1, &s,
				(float_t) 0, w_->length());
		s = sqrtf(s / (w_->length() - 1));
		//LOG_DEBUG("[%.8f, %.8f, %.8f, %.8f]", m, s, wm, ws);
		float_t rand = min(s / ws, (float_t) 1);
		if (rand < 0.7) {
			w_->set_pdata(0, i);
			count += 1;
		} else
			w_->upgrade_index()->push_back(i);
	}
	LOG_DEBUG("[%d / %d / %d] [%.8f, %.8f]", count, w_->num() - count,
			w_->num(), wm, ws);
	//w_->blob_size();
}

void train_net() {
	int batch_size = 100;

	int max_iter = 120000;

	int test_iter = 500;
	int train_test_iter = 1000;

	//int *train_lr{5000, 150000, 300000};

	//set gpu device if training by gpu
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif
#endif
	//set random seed
	set_rand_seed();

//log output
	std::ofstream logger(
			"/home/seallhf/4T/dataset/cifar/resnet_20_p.log",
			ios::binary);
	logger.precision(std::numeric_limits<float_t>::digits10);

	//log output
	std::ofstream precious_logger(
			"/home/seallhf/4T/dataset/cifar/resnet_precious_20_p.log",
			ios::binary);
	precious_logger.precision(std::numeric_limits<float_t>::digits10);
	//log output
	std::ofstream precious_train_logger(
			"/home/seallhf/4T/dataset/cifar/resnet_train_precious_20_p.log",
			ios::binary);
	precious_train_logger.precision(std::numeric_limits<float_t>::digits10);

	std::ofstream center_logger(
			"/home/seallhf/4T/dataset/cifar/resnet_center_20_p.log",
			ios::binary);
	center_logger.precision(std::numeric_limits<float_t>::digits10);

	//network *net = create_mobilenet(batch_size,train);
	//network *net = create_vgg_16_net(batch_size, train);
//	network *net = create_resnet_tall(batch_size, 10, train);
//	create_mobilenet(batch_size,train);
	network *net = create_res20net(batch_size, train);
//	net->load_weights("/home/seallhf/4T/cacue/cifar/resnet_30000_20_fatx1_pruneable_channel_rand_07_bn_upgrade_history_8w.model");
//	net->weights_pre_proc(weight_initial);
//	net->upgrade_bn_initial();

	//network *net = create_res32net_fat(batch_size,train,1);
//	blobs *inputs = new blobs();
//	inputs->push_back(new blob(batch_size, 3, 28, 28, 0, train));
//
//	bin_blob *label_ = new bin_blob(batch_size, 1, 1, 1, 0, test);
//	network *net = phrase_network("/home/seallhf/experiments/pruned/res20_pruned.ch", inputs);
//	net->set_is_use_bias(false);
//	net->delete_op(net->op_count() - 1);
//
//	blobs *labels = new blobs();
//	labels->push_back(net->output_blob());
//	labels->push_back(label_);
//	softmax_with_loss_op *loss = new softmax_with_loss_op(labels);
//	net->push_back_op(loss);
//
//	net->load_weights("/home/seallhf/experiments/pruned/res20_pruned.model");
//	net->set_weights_type(msra);
	//net->check();

	sgd_solver *sgd = new sgd_solver(net);
	sgd->set_positive_regularize(false);
	sgd->set_train_stage(STD_STAGE);

	float_t alpha = 0.7;
	//sgd->reinitialize();

	sgd->set_pruneable_alpha(alpha);

	sgd->set_lr(0.1);
	sgd->set_weight_decay(0.0001);
//	sgd->set_pruneable_decay(0.0004);
	sgd->set_regularize(cacu::regularize_type::L2);
//	sgd->set_pruneable_lr(0.1);

	string datapath = "/home/seallhf/4T/dataset/cifar/32x32_bmp/train/";
	string trainlist = "/home/seallhf/4T/dataset/cifar/train_list_norm.txt";
	string valdatapath = "/home/seallhf/4T/dataset/cifar/32x32_bmp/val/";
	string vallist = "/home/seallhf/4T/dataset/cifar/test_list.txt";
	string meanfile = "/home/seallhf/4T/dataset/cifar/mean.binproto";

	vector<string> full_data;

	vector<string> full_data_test;
	vector<vec_i> full_label_test;

	vector<string> vec;
	vec_i label(1);
	/**
	 * load mean data
	 */
	blob *mean_ = cacu_allocator::create_blob(1, 3, 32, 32, test);
	float_t *mean_data;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	imageio_utils::load_mean_file_gpu(mean_->s_data(), meanfile);
	mean_data = mean_->s_data_cpu();
#endif
#else
	imageio_utils::load_mean_file(mean_->s_data(),meanfile);
	mean_data = mean_->s_data();
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
	random_shuffle(full_data.begin(), full_data.end());

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
	blob *input_data = (blob*) net->input_blobs()->at(0);	//inputs->asblob(0);
	bin_blob *input_label = (bin_blob*) net->input_blobs()->at(1);	//label_;

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

	int previous_count = 0;
	int allcount = 0;

	vec_t channels(3);
//	channels[0] = 102.9801;
//	channels[1] = 115.9465;
//	channels[2] = 122.7717;
	channels[0] = -1;
	channels[1] = -1;
	channels[2] = -1;

	vec_i compare_label(batch_size);
	random_shuffle(full_data.begin(), full_data.end());
	for (int i = 1; i <= max_iter; ++i) {

		if (i % test_iter == 0) {
			count = 0;
			gettimeofday(&start, NULL);
			for (int test_c = 0; test_c < 100; test_c++) {
				for (int j = 0; j < batch_size; ++j) {
					if (step_index_test == ALL_TEST_DATA_SIZE)
						step_index_test = 0;
					//load image data
					//readdata(full_data_test[step_index_test].c_str(),input_data->p_data(j), mean_->s_data());
					//clipreaddata(full_data_test[step_index_test].c_str(),input_data->p_data(j), NULL);
					imageio_utils::center_clip_imread_channel_wise_norm(input_data->p_data(j),full_data_test[step_index_test].c_str(), 28, 28, channels);
//					imageio_utils::center_clip_imread_element_wise_norm(
//							input_data->p_data(j),
//							full_data_test[step_index_test].c_str(), 28, 28,
//							mean_data);
					//imageio_utils::imread_gpu(input_data->p_data(j),full_data_test[step_index_test].c_str(),input_data->length());
					input_label->copy2data(full_label_test[step_index_test], j);
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

			LOG_INFO("test accuracy: %f", (float_t )count / batch_size / 100);
			precious_logger << count << endl;
			precious_logger.flush();
		}
		if (i % train_test_iter == 0) {
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
				imageio_utils::clip_imread_channel_wise_norm(input_data->p_data(j),(datapath + vec[0]).c_str(), 28, 28, channels);
//				imageio_utils::clip_imread_element_wise_norm(
//						input_data->p_data(j), (datapath + vec[0]).c_str(), 28,
//						28, mean_data);
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
				vec = split(file_, " ");
				label[0] = strtoul(vec[1].c_str(), NULL, 10);
				//load image data
				//clipreaddata((datapath + vec[0]).c_str(), input_data->p_data(j), mean_->s_data());
				//clipreaddata((datapath + vec[0]).c_str(), input_data->p_data(j), NULL);
				imageio_utils::clip_imread_channel_wise_norm(input_data->p_data(j),(datapath + vec[0]).c_str(), 28, 28, channels);
//				imageio_utils::clip_imread_element_wise_norm(
//						input_data->p_data(j), (datapath + vec[0]).c_str(), 28,
//						28, mean_data);
				//cacu_print(input_data->p_data(j), 10);
				input_label->copy2data(label, j);
				step_index += 1;
			}
			//net->predict();
			sgd->train_iter(i);

//			blob *mean =
//					net->get_ops<batch_normalize_op>()[0]->get_storage_blob(2);
//			mean->blob_size();
//			blob *var = net->get_ops<batch_normalize_op>()[0]->get_storage_blob(
//					3);
//			blob *std = net->get_ops<batch_normalize_op>()[0]->get_storage_blob(
//					4);

//			weight *mean = net->get_ops<batch_normalize_op>()[0]->get_weight(0);
//			weight *var = net->get_ops<batch_normalize_op>()[0]->get_weight(1);
//
//			cacu_print("mean", mean->s_data(), mean->count());
//			cacu_print("var", var->s_data(), var->count());

			gettimeofday(&end, NULL);
		}

		if (i % 10 == 0) {
			diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec
					- start.tv_usec;
			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, sgd->lr(), diff / 1000);
			((softmax_with_loss_op*) net->get_op(net->op_count() - 1))->echo();
			//net->get_op<center_loss_op>(net->op_count() - 3)->echo();
		}
		logger
				<< ((softmax_with_loss_op*) net->get_op(net->op_count() - 1))->loss()
				<< endl;
		logger.flush();

//		center_logger << (net->get_op<center_loss_op>(net->op_count() - 3))->loss()
//						<< endl;
//		center_logger.flush();

//		if(i < 40000){
//			if(i % 30001 / 15000 == 1)
//				sgd->set_train_stage(UPGRADE_STAGE);
//			else if(i % 30001 / 15000 == 2)
//				sgd->set_train_stage(STD_STAGE);
//		}
//		else if(i < 90000 && i >= 40000){
//			if((i - 40000) % 40001 / 20000 == 1)
//				sgd->set_train_stage(UPGRADE_STAGE);
//			else if((i - 40000) % 40001 / 20000 == 2)
//				sgd->set_train_stage(STD_STAGE);
//		}
//		else{
//			if((i - 90000) % 40000 == 0)
//				sgd->set_train_stage(UPGRADE_STAGE);
//		}

		if (i % 40000 == 0) {
			sgd->set_lr_iter(0.1);
			//sgd->set_train_stage(PRUNE_STAGE);
		}
//		if (i % 80000 == 0){
//		//if (i == 100000 || i == 150000)
//			sgd->set_lr_iter(0.1);
//			sgd->set_train_stage(PRUNE_STAGE);
//			net->get_op<center_loss_op>(net->op_count() - 3)->set_loss_weight(net->get_op<center_loss_op>(net->op_count() - 3)->loss_weight() * 0.1);
//		}

		if (i % 10000 == 0) {
			ostringstream oss;
			oss << "/home/seallhf/4T/dataset/cifar/resnet_20_" << i
					<< "_p.model";
			net->save_weights(oss.str());
		}
	}

	logger.close();
	precious_logger.close();
	precious_train_logger.close();
	center_logger.close();

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
