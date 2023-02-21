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

#ifndef TRAIN_TN_CIFAR_HPP_
#define TRAIN_TN_CIFAR_HPP_


#include "../../tools/time_utils.h"
#include "../../cacu/cacu.h"

#include "../../cacu/solvers/solvers.h"

#include "../../tools/string_utils.h"

#include "generate_models_cifar.h"
#include "../cifar10/data_proc.h"

using namespace cacu;

void train_net() {
	int batch_size = 100;

	int max_iter = 50000;

	int test_iter = 100;
	int train_test_iter = 100;

	string root_path = "/home/seallhf/4T/dataset/cifar/";

	//set gpu device if training by gpu
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif
#endif
	//set random seed
	set_rand_seed();

	//log output
	std::ofstream logger(root_path + "res50netg.log",
			ios::binary);
	logger.precision(std::numeric_limits<cacu::float_t>::digits10);

	//log output
	std::ofstream precious_logger(root_path +
			"res50netd_.log",
			ios::binary);
	precious_logger.precision(std::numeric_limits<cacu::float_t>::digits10);
	//log output
	std::ofstream precious_train_logger(root_path +
			"res50net_train_precious.log",
			ios::binary);
	precious_train_logger.precision(std::numeric_limits<cacu::float_t>::digits10);

	std::ofstream p_logger(root_path + "train_precision_positive.txt", ios::binary);
	p_logger.precision(std::numeric_limits<cacu::float_t>::digits10);

	/*
	 * create generative model
	 */

	blob *random_z_ = cacu_allocator::create_blob(batch_size, 100, 1, 1, train);
	blob *mhead = cacu_allocator::create_blob(batch_size, 3, 32, 32, train);
	blob *dhead = cacu_allocator::create_blob(batch_size, 100, 1, 1, train);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1, train);

	network *gnet = new network();
	network *mnet = new network();
	network *dnet = new network();

	layer_block *generator = create_generator_32(random_z_, train);
	*gnet << generator;
//	gnet->load_weights("/home/seallhf/4T/dataset/cifar/models/generator_20000.model");

	blob *goutput = generator->oblob();

	layer_block *mapping_net = create_discriminator_32(mhead, train);
	*mnet << mapping_net;

	blob *moutput = mapping_net->oblob();//gnet->get_op<batch_normalize_op>(gnet->op_count() - 2)->out_data<blob>();

	layer_block *discriminator = create_discriminator(dhead, label_, train);
//	discriminator->load_weights("/home/seallhf/4T/dataset/cifar/models/discriminator_10000.model");
	*dnet << discriminator;

//	vector<convolution_op *> convs = gnet->get_ops<convolution_op>();
//	for(int i = 0; i < convs.size(); ++i){
//		weight *w = convs[i]->get_weight(0);
//		w->switch_body();
//		w->switch_channel();
//	}

	adam_solver *gsgd = new adam_solver(gnet);
	gsgd->set_regularize(cacu::regularize_type::L2);
//	gsgd->load_param("/home/haofang/experiment/generative/g_solver.txt");
	gsgd->set_beta1(0.5);
	gsgd->set_beta2(0.999);
	gsgd->set_lr(0.0000001);
	gsgd->set_weight_decay(0.01);

	sgd_solver *msgd = new sgd_solver(mnet);
	msgd->set_regularize(cacu::regularize_type::L2);
//  dsgd->load_param("/home/haofang/experiment/generative/d_solver.txt");
	msgd->set_lr(0.01);
	msgd->set_weight_decay(0.004);

	sgd_solver *dsgd = new sgd_solver(dnet);
	dsgd->set_regularize(cacu::regularize_type::L2);
//  dsgd->load_param("/home/haofang/experiment/generative/d_solver.txt");
	dsgd->set_lr(0.001);
	dsgd->set_weight_decay(0.004);

	string datapath = "/home/seallhf/4T/dataset/cifar/";
	string meanfile = "/home/seallhf/4T/dataset/cifar/mean.binproto";

	vector<vec_t> full_data;
	vector<vec_i> full_label;

	load_data(root_path, full_data, full_label);

	vector<vec_t> full_data_test;
	vector<vec_i> full_label_test;
	load_test_data(root_path, full_data_test, full_label_test);

	vec_i _full_label;
	for(int i = 0; i < full_label.size(); ++i)
		_full_label.push_back(full_label[i][0]);

	vec_i _full_label_test;
	for(int i = 0; i < full_label_test.size(); ++i)
		_full_label_test.push_back(full_label_test[i][0]);
	//load_data(datapath, full_data, full_label);

	int ALL_DATA_SIZE = full_data.size();
	/**
	 * read data for training
	 */
	blob *input_data = mhead;
	bin_blob *input_label = label_;

	/**
	 * read data for testing
	 */
	blob *output_data = dnet->output_blob();

	int step_index = 0;
	int step_index_test = 0;
	int step_index_train = 0;
	
	time_utils *timer = new time_utils();

	unsigned int max_index;
	cacu::float_t count = 0;

	int previous_count=0;
	int allcount = 0;

	cacu::float_t dloss1 = 0,dloss2 = 0,gloss = 0;

	softmax_with_loss_op *loss = dnet->get_ops<softmax_with_loss_op>()[0];

	blobs *indatas = new blobs((blob_base *&)mhead, (blob_base *&)goutput);
	abse_loss_op *loss_s = new abse_loss_op(indatas);

	//random_shuffle(full_data.begin(), full_data.end());
	for (int i = 1; i <= max_iter; ++i) {

		timer->start();
		for (int j = 0 ; j < batch_size ; ++j)
		{
			if (step_index == kCIFARDataCount)
				step_index = 0;
			input_data->copy2data(full_data[step_index], j);
			input_label->copy2data(full_label[step_index],j);
			step_index += 1;
		}

		mnet->set_phase(train);
		mnet->forward_propagate();
		random_z_->copy2data(moutput);
		dhead->copy2data(moutput);

		dnet->forward_propagate();
		dloss1 = loss->loss();
		dnet->backward_propagate();

		gnet->forward_propagate();

		loss_s->infer();
		gloss = loss_s->loss();
		loss_s->derivative();

		gnet->set_update_weight(false);
		gnet->backward_propagate();

		moutput->copy2diff(random_z_);
//		cacu_saxpby(dhead->s_diff(), 1, moutput->s_diff(), 1, dhead->count());
		mnet->backward_propagate();

//		gsgd->updates(i);
		msgd->updates(i);
		dsgd->updates(i);

//		cacu_print(blob_->s_data(), 100);

//		dsgd->train_iter(i);


//		dsgd->train_iter(i);

		//cacu_print(gnet->get_op(0)->get_weight(0)->s_diff(),10);
		//*/
		timer->end();

		if (i % 1 == 0) {
			
			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, dsgd->lr(), timer->get_time_span() / 1000);
			LOG_INFO("loss : %f", dloss1);
			LOG_INFO("loss_s : %f", gloss);
			//LOG_INFO("generator loss : %f", gloss);
		}

		precious_logger << dloss1<< endl;
		precious_logger.flush();


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

				mnet->predict();
				dhead->copy2data(moutput);
				dnet->predict();

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

			ostringstream oss;
			random_z_->copy2data(moutput);
			gnet->forward_propagate();
			oss << root_path << "generative/" << "test_gpu_" << i << ".jpg";
			imageio_utils::imwrite(((blob*)generator->get_oblob()),oss.str().c_str());

			LOG_INFO("precision: %f", count);
		}

		if (i % 20000 == 0){
			msgd->set_lr_iter(0.1);
			dsgd->set_lr_iter(0.1);
		}

		if (i % 10000 == 0) {
			ostringstream oss;
			oss.str("");
			oss << root_path << "models/" << "single_discriminator_" << i << ".model";
			discriminator->save_weights(oss.str());

			oss.str("");
			oss << root_path << "models/" << "mappingnet_" << i << ".model";
			mapping_net->save_weights(oss.str());
		}
	}
	logger.close();
	precious_logger.close();
	precious_train_logger.close();

	LOG_INFO("optimization is done!");

	vector<vec_t>().swap(full_data);
	vector<vec_i>().swap(full_label);
	delete gnet;
	delete dnet;
	delete dsgd;
	delete timer;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif
