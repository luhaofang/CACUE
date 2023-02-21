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

#include "cifar_quick_net.h"
//#include "cifar_test_net.h"
#include "data_proc.h"
#include "../../tools/vec_utils.h"

using namespace cacu;
using namespace cacu_tools;

//upgrade training for initial convolution & fc weights
void weight_initial(weight *&w_){
	if(w_->length() == 1)
		return;
	vec_t temp(w_->length());
	vec_t wtemp(w_->count());
	float wm = 0, ws = 0;
	float_t alpha = 0.86;
	cacu_sumbysize(BYWIDTH, w_->s_data(), w_->count(), (float_t)1, &wm, (float_t)0, w_->count());
	wm = wm / w_->count();
	cacu_sdxsize(w_->s_data(), w_->count(), (float_t)-1 * wm, (float_t)1, &wtemp[0]);
	cacu_sqr(&wtemp[0], w_->count(), &wtemp[0]);
	cacu_sumbysize(BYWIDTH, &wtemp[0], w_->count(), (float_t)1, &ws, (float_t)0, w_->count());
	ws = sqrtf(ws / (w_->count() - 1));

	cacu_copy(w_->s_data(), w_->count(), &wtemp[0]);
//	w_->set_init_type(gaussian, ws * alpha);
//	w_->set_init_type(constant);

	for(int i = 0; i < w_->num(); ++i){
		float m = 0, s = 0;
		cacu_sumbysize(BYWIDTH, &wtemp[i * w_->length()], w_->length(), (float_t)1, &m, (float_t)0, w_->length());
		m = m / w_->length();
		cacu_sdxsize(&wtemp[i * w_->length()], w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
		cacu_sqr(&temp[0], w_->length(), &temp[0]);
		cacu_sumbysize(BYWIDTH, &temp[0], w_->length(), (float_t)1, &s, (float_t)0, w_->length());
		s = sqrtf(s / (w_->length() - 1));
		LOG_DEBUG("[%.8f, %.8f, %.8f, %.8f]", m, s, wm, ws);
//		if(s >= ws / w_->length()){
//			cacu_copy(&wtemp[i * w_->length()], w_->length(), w_->p_data(i));
//			LOG_DEBUG("copy: %d", i);
//		}
		if(s < ws / w_->length()){
//			gaussian_init(w_->p_data(i), w_->length(), wm, ws * alpha);
			w_->set_pdata(0, i);
			LOG_DEBUG("copy: %d", i);
		}
		else
			w_->upgrade_index()->push_back(i);
	}

	//shortcut silent weight
//	for(int i = 0; i < w_->num(); ++i){
//		float_t m = 0, s = 0;
//#if __USE_DEVICE__ == ON
//		cacu_sumbysize_cpu(BYWIDTH, w_->p_data_cpu(i, false), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
//#else
//		cacu_sumbysize(BYWIDTH, w_->p_data(i), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
//#endif
//		m = m / w_->length();
//#if __USE_DEVICE__ == ON
//		cacu_sdxsize_cpu(w_->p_data_cpu(i, false), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
//#else
//		cacu_sdxsize(w_->p_data(i), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
//#endif
//		cacu_sqr_cpu(&temp[0], w_->length(), &temp[0]);
//		cacu_sumbysize_cpu(BYWIDTH, &temp[0], w_->length(), (float_t)1, &s, (float_t)0, w_->length());
//		s = sqrtf(s / (w_->length() - 1));
//		float_t rand = min(s / ws, (float_t)1);
//		LOG_DEBUG("[%.8f, %.8f, %.8f, %.8f, %.8f]", m, s, wm, ws, rand);
//		if(rand < alpha)
//			w_->set_pdata(0, i);
//		else
//			w_->upgrade_index()->push_back(i);
//	}
	w_->blob_size();
}

void weight_print(weight *&w_){
	if(w_->length() == 1)
		return;
	w_->blob_size();
	for(int i = 0; i < w_->num(); ++i)
		if(!FIND_FROM_VEC(*w_->upgrade_index(), i)){
			for(int j = 0 ; j < 3 ; ++j)
				cout << w_->p_data(i)[j] << ",";
		}
	cout << endl;
}

void weight_diff_recode(weight *&w_, std::ostream &os){
	if(w_->length() == 1)
		return;
	vec_t temp(w_->length());
	vec_t wtemp(w_->count());
	w_->update_index()->clear();
	int count = 0;
	float_t acc_cdf = 0;
	float_t rand;
	int sign = 0;
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

		rand = gaussrand(s / sd, md);
		sign = 1;
		acc_cdf += (md >= m) ? (1. - cacu_normal_cdf(m, sd, md)) / w_->num() :
				(cacu_normal_cdf(m, sd, md)) / w_->num();
		if((md >= m && rand < m) || (md < m && rand >= m)){
			count += 1;
			w_->update_index()->push_back(i);
			sign = 0;
		}
		acc_sd += sd / w_->num();
	}
	os << acc_sd << ",";
}

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

//	network *net = create_cifar_upgrade_net(batch_size,train);
//	network *net = create_cifar_quick_net(batch_size,train);
//	net->load_weights("/Users/seallhf/Documents/datasets/cifar10/cifar10_quick_test.model");

//	blobs *inputs = new blobs();
//	inputs->push_back(new blob(batch_size, 3, 32, 32, 0, train));
//	//inputs->push_back(new bin_blob(batch_size, 1, 1, 1, 0, train));
//	bin_blob *label_ = new bin_blob(batch_size, 1, 1, 1, 0, train);
//	network *net = phrase_network("/Users/seallhf/Documents/datasets/cifar10/cifar10_quick_test_pruned.ch", inputs);
//	net->set_is_use_bias(false);
////	net->set_is_use_shift(false);
//	vector<convolution_op*> ops = net->get_ops<convolution_op>();
////
//	ops[0]->set_weight_init_type(gaussian, 0.0001);
//	ops[1]->set_weight_init_type(gaussian, 0.01);
//	ops[2]->set_weight_init_type(gaussian, 0.01);
//
//	vector<inner_product_op*> ips = net->get_ops<inner_product_op>();
//	ips[0]->set_weight_init_type(gaussian, 0.1);
//	ips[1]->set_weight_init_type(gaussian, 0.1);
//
//	net->delete_op(net->op_count() - 1);
//	blobs *labels = new blobs();
//	labels->push_back(net->output_blob());
//	labels->push_back(label_);
//	softmax_with_loss_op *loss = new softmax_with_loss_op(labels);
//	net->push_back_op(loss);
//
//	//net->load_weights_from("/home/haofang/experiment/cifar10/cifar10_quick_normal.model",10);
//	net->load_weights("/Users/seallhf/Documents/datasets/cifar10/cifar10_quick_test_pruned_wg.model");
//	net->weights_pre_proc(weight_initial);
//	net->upgrade_bn_initial();

	float_t alpha_decay = 0.85;
	sgd_solver *sgd = new sgd_solver(net);
	sgd->set_regularize(cacu::regularize_type::L2);
//	sgd->set_positive_regularize(true);
//	sgd->set_is_pruning(true);
	sgd->set_pruneable_alpha(alpha_decay);
//	sgd->set_is_upgrade_optimize(true);
	sgd->set_train_stage(STD_STAGE);

	//no L1 regularization
//	sgd->set_pruneable_decay(0.1);
//	gsgd->load_param("/home/haofang/experiment/generative/g_solver.txt");
//	sgd->set_alpha(0.5);
//	sgd->set_beta(0.9);
	sgd->set_lr(0.001);
	sgd->set_weight_decay(0.004);
//	sgd->set_pruneable_decay(0.0004);


	string rootpath = "/home/seallhf/4T/dataset/cifar/";
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

//	blob *input_data = inputs->asblob(0);//(blob*)net->input_blobs()->at(0);
//	bin_blob *input_label = label_;//(bin_blob*)net->input_blobs()->at(1);

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

//		for(int k = 0 ; k < net->op_count(); ++k)
//			sgd->data_recode(weight_diff_recode, diff_log, k);

		diff_log << endl;

		sgd->updates(i);

		timer->end();

		if(i % 10 == 0){

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


//		if(i% 10000 == 0){
//			if(i / 10000 == 1){
//				sgd->set_train_stage(UPGRADE_STAGE);
//			}
//			if(i / 10000 == 2)
//				sgd->set_train_stage(STD_STAGE);
//		}

//		if(i % 7000 == 0)
//			sgd->set_pruneable_alpha(alpha_decay - 0.2 * i / 7000);

		if(i % 4000 == 0){
			sgd->set_lr_iter(0.1);
		}
//		if(i % 10000 == 0){
//			ostringstream oss;
//			oss << rootpath << "cifar10_quick_test" << i / 10000 << ".model";
//			net->save_weights(oss.str());
////			net->get_op<center_loss_op>(net->op_count() - 3)->set_loss_weight(
////						net->get_op<center_loss_op>(net->op_count() - 3)->loss_weight() * 2);
//		}

//		if(i % 7000 == 0){
//			ostringstream oss;
//			oss << rootpath << "cifar10_quick_test" << i / 7000 << ".model";
//			net->save_weights(oss.str());
//		}
	}
	LOG_INFO("optimization is done!");
	net->save_weights(rootpath + "cifar10_quick_test.model");

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

