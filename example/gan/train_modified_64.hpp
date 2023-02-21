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

#ifndef TRAIN_MODIFIED_HPP_
#define TRAIN_MODIFIED_HPP_

#include "../../cacu/cacu.h"

#include "../../cacu/solvers/solvers.h"
//#include "../../cacu/solvers/sgd_solver.h"

#include "../../tools/string_utils.h"
#include "../frcnn/tools_frcnn.h"

#include "../imagenet/vgg_net.h"
#include "../imagenet/data_proc.h"
#include "../imagenet/resnet_18.h"
#include "../imagenet/resnet_50.h"
#include "../imagenet/mobilenet.h"
#include "generate_models.h"

using namespace cacu;

void train_net() {
	int batch_size = 64;

	int max_iter = 100000;

	int test_iter = 100;
	int train_test_iter = 100;

	string root_path = "/home/seallhf/4T/dataset/celeba/data/";

	//set gpu device if training by gpu
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif
#endif
	//set random seed
	set_rand_seed();

	//log output
	std::ofstream logger(root_path + "res50netg_64_sn.log", ios::binary);
	logger.precision(std::numeric_limits<cacu::float_t>::digits10);

	//log output
	std::ofstream precious_logger(root_path + "res50netd_64_sn.log", ios::binary);
	precious_logger.precision(std::numeric_limits<cacu::float_t>::digits10);
	//log output
	std::ofstream precious_train_logger(
			root_path + "res50net_train_precious.log", ios::binary);
	precious_train_logger.precision(
			std::numeric_limits<cacu::float_t>::digits10);

	/*
	 * create generative model
	 */
	blobs *gin_blobs = new blobs();
	blobs *din_blobs = new blobs();
	blob *random_z_ = cacu_allocator::create_blob(batch_size, 100, 1, 1, train);
	blob *blob_ = cacu_allocator::create_blob(batch_size, 3, 64, 64, train);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1,
			train);
	din_blobs->push_back(blob_);
	din_blobs->push_back(label_);
	gin_blobs->push_back(random_z_);

	blob *noise_blob_ = cacu_allocator::create_blob(batch_size, 3, 64, 64, test);

	network *gnet = new network(gin_blobs);

	network *dnet = new network(din_blobs);

	layer_block *generator = create_generator_64(random_z_, 128, train);
//	generator->load_weights("/home/seallhf/4T/dataset/celeba/data/models/generator_70000_64_sn.model");

	blob *suspicious = (blob*) generator->get_oblob();

	layer_block *discriminator = create_discriminator_sn_64(blob_, 128, label_,
			train); //create_mobilenet(batch_size,train);//create_alexnet(batch_size,train);
//	discriminator->load_weights("/home/seallhf/4T/dataset/celeba/data/models/discriminator_70000_64_sn.model");
	//discriminator->set_update_weight(false);
	//discriminator->layers(0)->get_head_op()->__NEED_BACK_PROPAGATE__(true);
	//layer_block *discriminator = create_vgg_16_block(blob_, label_, train);//create_mobilenet(batch_size,train);//create_alexnet(batch_size,train);
	//discriminator->set_update_weight(false);

	*gnet << generator;
	*dnet << discriminator;

//	string modelpath = "/home/haofang/experiment/imagenet/models/res50.model";
	//dnet->load_weights_from(modelpath,0,dnet->op_count() - 2);
//	discriminator->load_weights("/home/seallhf/4T/dataset/celeba/data/models/discriminator_3000_64.model");//net->load_weights("/home/seal/4T/cacue/imagenet/alex_net_20000.model");

	adam_solver *gsgd = new adam_solver(gnet);
	gsgd->load_param(root_path + "g_solver.txt");
	gsgd->set_train_stage(STD_STAGE);

	adam_solver *dsgd = new adam_solver(dnet);
	dsgd->load_param(root_path + "d_solver.txt");
	dsgd->set_train_stage(STD_STAGE);

	//string datapath = "/home/haofang/experiment/imagenet/../../data/pascalvoc/VOCdevkit/VOC2012/objects/";
	//string trainlist = "/home/haofang/experiment/imagenet/../../data/pascalvoc/VOCdevkit/VOC2012/object_list.txt";
	string datapath = root_path + "64x64/";
	string trainlist = root_path + "train_list_bmp.txt";

	vector<string> full_data;

	vector<string> vec;
	vec_i label(batch_size);

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

	int ALL_DATA_SIZE = full_data.size();
	/**
	 * read data for training
	 */
	blob *input_data = blob_;
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

	int previous_count = 0;
	int allcount = 0;
	cacu::float_t dloss1 = 0, dloss2 = 0, gloss = 0;

	int n_s = 1;
	int noise_s = 4;

	sigmoid_with_loss_op* loss_op = dnet->get_op<sigmoid_with_loss_op>(
			dnet->op_count() - 1);
//	hinge_loss_op* loss_op = dnet->get_op<hinge_loss_op>(
//				dnet->op_count() - 1);
	random_shuffle(full_data.begin(), full_data.end());
	for (int i = 1; i <= max_iter; ++i) {

		{
			timer->start();
			for (int j = 0; j < batch_size; ++j) {
				if (step_index == ALL_DATA_SIZE) {
					step_index = 0;
					random_shuffle(full_data.begin(), full_data.end());
				}
				file_ = full_data[step_index];
				//vec = split(file_, "_");
				//label[j] = strtoul(split(vec[1],".")[0].c_str(), NULL, 10);
				//load image data
				readimg(blob_->p_data(j), (datapath + file_).c_str());

				step_index += 1;
			}
			//train discriminator by true image
			label_->set_data(1);
			//loss_op->set_loss_weight(0.5);
//			dsgd->train_iter(i);
			dnet->set_phase(train);
			dnet->forward_propagate();
			dnet->backward_propagate();
			dnet->input_blobs()->asblob(0)->_RESET_DIFF();

			dloss1 = loss_op->loss();

			//generate the fake image
			random_z_->set_init_type(uniform, 1);
			gnet->forward_propagate();
			cacu_copy(gnet->output_blob()->s_data(),
					gnet->output_blob()->count(), blob_->s_data());
//			if(i % noise_s == 0){
//				noise_blob_->set_init_type(gaussian, 0.036);
//				cacu_saxpby(noise_blob_->s_data(), 1., blob_->s_data(), 1., noise_blob_->count());
//			}
			//cacu_print(blob_->s_data(),10);

			//train discriminator by fake label
			label_->set_data(0);
			dsgd->train_iter(i);
			dloss2 = loss_op->loss();

			//train generator by zero label
			//input_label->copy2data(label);
			cacu_copy(blob_->s_diff(), blob_->count(),
					gnet->output_blob()->s_diff());
			//cacu_print(blob_->s_diff(),10);

			gsgd->update_direction(maximize);
			gnet->backward_propagate();
			if (i % n_s == 0)
				gsgd->updates(i / n_s);
		}
		logger << dloss2 << endl;
		logger.flush();

		precious_logger << dloss1 + dloss2 << endl;
		precious_logger.flush();

		timer->end();

		if (i % 1 == 0) {

			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, dsgd->lr(),
					timer->get_time_span() / 1000);
			LOG_INFO("discriminator loss : %f", (dloss1 + dloss2));
			LOG_INFO("generator loss : %f", dloss2);
			//LOG_INFO("generator loss : %f", gloss);
		}

		if (i % 40000 == 0) {
			dsgd->set_lr_iter(0.1);
			gsgd->set_lr_iter(0.1);
		}

		if (i % 100 == 0) {
			ostringstream oss;
			random_z_->set_init_type(uniform, 1);
			gnet->predict();
			oss << root_path << "generative_64_sn/" << "test_" << i << "_64.jpg";
			imageio_utils::imwrite(((blob*) generator->get_oblob()),
					oss.str().c_str());
		}
		if (i % 1000 == 0) {
			ostringstream oss;

			oss << root_path << "models/" << "generator_" << i << "_64_sn.model";
			generator->save_weights(oss.str());
			oss.str("");
			oss << root_path << "models/" << "discriminator_" << i
					<< "_64_sn.model";
			discriminator->save_weights(oss.str());
			oss.str("");
		}
	}

	logger.close();
	precious_logger.close();
	precious_train_logger.close();

	LOG_INFO("optimization is done!");

	vector<string>().swap(full_data);
	delete gnet;
	delete gsgd;
	delete dnet;
	delete dsgd;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}

#endif
