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

#ifndef TRAIN_BN_FACE_CGAN_HPP_
#define TRAIN_BN_FACE_CGAN_HPP_


#include "../../tools/time_utils.h"
#include "../../cacu/cacu.h"

#include "../../cacu/solvers/solvers.h"

#include "../../tools/string_utils.h"

#include "cycle_models.h"
#include "../imagenet/data_proc.h"

using namespace cacu;

void train_net() {
	int batch_size = 8;

	int max_iter = 200000;

	int test_iter = 100;
	int train_test_iter = 100;

	string root_path = "/home/seallhf/4T/dataset/";

	string dir_path = "/home/seallhf/cgan/";

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
			"res50netd.log",
			ios::binary);
	precious_logger.precision(std::numeric_limits<cacu::float_t>::digits10);
	//log output
	std::ofstream precious_train_logger(root_path +
			"res50net_train_precious.log",
			ios::binary);
	precious_train_logger.precision(std::numeric_limits<cacu::float_t>::digits10);


	/*
	 * create generative model
	 */

	network *gnet = new network();
	network *dnet = new network();

	layer_block *generator = create_generator_cycleface_bn(batch_size, 64, train);
	layer_block *discriminator = create_discriminator_cycleface_bn(batch_size, 128, train);

	*gnet << generator;
//	generator->load_weights(root_path + "models/" + "generator_400000_good.model");
	*dnet << discriminator;
//	discriminator->load_weights(root_path + "models/" + "discriminator_400000_good.model");

	adam_solver *gsgd = new adam_solver(gnet);
	gsgd->set_regularize(cacu::regularize_type::L2);
//	gsgd->load_param("/home/haofang/experiment/generative/g_solver.txt");
	gsgd->set_beta1(0.);
	gsgd->set_beta2(0.9);
	gsgd->set_lr(0.0001);
	gsgd->set_weight_decay(0.0001);


	adam_solver *dsgd = new adam_solver(dnet);
	dsgd->set_regularize(cacu::regularize_type::L2);
//  dsgd->load_param("/home/haofang/experiment/generative/d_solver.txt");
	dsgd->set_beta1(0.);
	dsgd->set_beta2(0.9);
	dsgd->set_lr(0.0001);
	dsgd->set_weight_decay(0.0001);

	string datapath = root_path;
	string trainlist = root_path + "HQ_facedata_list_128.txt";

	vector<string> full_data;

	vector<string> vec;
	vec_i label(1);

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

	int ALL_DATA_SIZE = full_data.size();

	blob *suspicious = (blob*)generator->get_oblob();
	blob *random_z_ = generator->layers(0)->get_head_op()->in_data<blob>();
//	bin_blob *glabel = gnet->input_blobs()->asbinblob(1);

	blob *dhead = discriminator->layers(0)->get_head_op()->in_data<blob>();
	blob *doutput= (blob*)discriminator->get_oblob();
//	bin_blob *dlabel = dnet->input_blobs()->asbinblob(1);\

	bin_blob *gan_label = dnet->get_ops<hinge_loss_op>()[0]->in_datas()->asbinblob(1);

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

	cacu::float_t dloss1 = 0, dloss2 = 0, gloss = 0;

	//LOG_DEBUG("fuck! %d", gan_datas->size());

	hinge_loss_op *loss_dis_s = dnet->get_ops<hinge_loss_op>()[0];

	int n_s = 5;
	int n_b = 20;

	random_shuffle(full_data.begin(), full_data.end());
	for (int i = 1; i <= max_iter; ++i) {
		timer->start();
		for(int b = 0; b < n_b; ++b){
			for (int j = 0 ; j < batch_size ; ++j)
			{
				if (step_index == ALL_DATA_SIZE) {
					step_index = 0;
					random_shuffle(full_data.begin(), full_data.end());
				}
				file_ = full_data[step_index];
				imageio_utils::center_clip_imread_element_wise_norm(dhead->p_data(j),(datapath + file_).c_str(), 128, 128, NULL);
				step_index += 1;
			}
			cacu_sdxsize(dhead->s_data(), dhead->count(), -127.5, 127.5, dhead->s_data());

			//train discriminator by true image
			dnet->set_update_weight(true);
			gan_label->set_data(1);
			//dnet->set_phase(train);
			dnet->forward_propagate();
			dnet->backward_propagate();
			//dsgd->updates(i);
			dloss1 = (loss_dis_s->loss()) / 2;

			//generate the fake image
			random_z_->set_init_type(uniform, 1);
			gnet->set_phase(test);
			gnet->forward_propagate();
			cacu_copy(gnet->output_blob()->s_data(), gnet->output_blob()->count(), dhead->s_data());

			//train discriminator by fake label
			gan_label->set_data(0);
			//dnet->set_phase(train);
			dnet->forward_propagate();
			dnet->backward_propagate();
			if(b == n_b - 1)
				dsgd->updates(i);
			dloss1 += (loss_dis_s->loss()) / 2;
		}
		if(i % n_s == 0){

			for(int b = 0; b < n_b; ++b){
				gnet->set_update_weight(true);
				//train generator by zero label
				//generate the fake image
				dhead->_RESET_DIFF();
				random_z_->set_init_type(uniform, 1);

				gnet->set_phase(train);
				gnet->forward_propagate();
				cacu_copy(gnet->output_blob()->s_data(), gnet->output_blob()->count(), dhead->s_data());

				dnet->set_update_weight(false);
				gan_label->set_data(0);
				//dnet->set_phase(train);

				dnet->forward_propagate();
				dnet->backward_propagate();
				//dsgd->updates(i);
				//dloss2 = loss_gen->get_op<mean_op>(1)->mean();
				dloss2 = (loss_dis_s->loss());

				cacu_copy(dhead->s_diff(), gnet->output_blob()->count(), gnet->output_blob()->s_diff());
				//gnet->set_phase(train);
				gsgd->update_direction(maximize);
				gsgd->net()->backward_propagate();
				if(b == n_b - 1)
					gsgd->updates(i / n_s);
				dnet->_RESET_WEIGHT_DIFF();
			}
		}

		//*/
		timer->end();


		if (i % 1 == 0) {

			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, dsgd->lr(), timer->get_time_span() / 1000);
			LOG_INFO("discriminator loss : %f", dloss1);
			LOG_INFO("generator loss : %f", dloss2);
			//LOG_INFO("generator loss : %f", gloss);

			logger << dloss2 << endl;
			logger.flush();

			precious_logger << dloss1 << endl;
			precious_logger.flush();
		}


		if (i % 6250 == 0){
			dsgd->set_lr_iter(0.1);
			gsgd->set_lr_iter(0.1);
		}

		if (i % 100 == 0) {
			ostringstream oss;

			oss << dir_path << "generative/" << "test_" << i << ".jpg";
			imageio_utils::imwrite(((blob*)generator->get_oblob()),oss.str().c_str());
		}
		if (i % 625 == 0) {
			ostringstream oss;

			oss << dir_path << "models/" << "generator_" << i << ".model";
			generator->save_weights(oss.str());
			oss.str("");
			oss << dir_path << "models/" << "discriminator_" << i << ".model";
			discriminator->save_weights(oss.str());
			oss.str("");
		}
	}

	ostringstream oss;
	oss << dir_path << "models/" << "generator.model";
	//generator->save_weights(oss.str());
	oss.str("");
	oss << dir_path << "models/" << "discriminator.model";
	//discriminator->save_weights(oss.str());
	logger.close();
	precious_logger.close();
	precious_train_logger.close();

	LOG_INFO("optimization is done!");

	vector<string>().swap(full_data);
	delete gnet;
	delete gsgd;
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
