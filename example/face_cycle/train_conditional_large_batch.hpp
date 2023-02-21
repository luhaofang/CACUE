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

#ifndef TRAIN_CONDITIONAL_LARGE_BATCH_CGAN_HPP_
#define TRAIN_CONDITIONAL_LARGE_BATCH_CGAN_HPP_


#include "../../tools/time_utils.h"
#include "../../cacu/cacu.h"

#include "../../cacu/solvers/solvers.h"

#include "../../tools/string_utils.h"

#include "cycle_models.h"
#include "models.h"
#include "../imagenet/data_proc.h"

using namespace cacu;


void train_net() {
	int batch_size = 8;

	int max_iter = 10000;

	int test_iter = 100;
	int train_test_iter = 100;

	string root_path = "/home/seallhf/face_transform/";
	string data_path = "/home/seallhf/cgan/";

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

	std::ofstream precious_constant_logger(root_path +
			"constant_loss.log",
			ios::binary);
	precious_constant_logger.precision(std::numeric_limits<cacu::float_t>::digits10);


	/*
	 * create generative model
	 */

	network *gnet = new network();
	network *dnet = new network();
	network *kpnet = new network();
	network *dznet = new network();

	layer_block *generator = create_generator_cycleface(batch_size, 64, train);
	layer_block *discriminator = create_discriminator_cycleface(batch_size, 128, train);
	layer_block *discriminator_randomz = create_discriminator_random_cycleface(batch_size, train);
	layer_block *discriminator_kp = create_discriminator_68regression_cycleface(batch_size, train);

	*gnet << generator;
	generator->load_weights(root_path + "models/" + "generator_370000.model");
	*dnet << discriminator;
	discriminator->load_weights(root_path + "models/" + "discriminator_370000.model");
	*kpnet << discriminator_kp;
	discriminator_kp->load_weights(root_path + "models/" + "regressor68_100000.model");
	*dznet << discriminator_randomz;
//	discriminator_randomz->load_weights(root_path + "models/" + "random_20000.model");
	kpnet->set_update_weight(false);

	adam_solver *gsgd = new adam_solver(gnet);
	gsgd->set_regularize(cacu::regularize_type::L2);
//	gsgd->load_param("/home/haofang/experiment/generative/g_solver.txt");
	gsgd->set_beta1(0.);
	gsgd->set_beta2(0.9);
	gsgd->set_lr(0.00001);
	gsgd->set_weight_decay(0.000);


	adam_solver *dsgd = new adam_solver(dnet);
	dsgd->set_regularize(cacu::regularize_type::L2);
//  dsgd->load_param("/home/haofang/experiment/generative/d_solver.txt");
	dsgd->set_beta1(0.);
	dsgd->set_beta2(0.9);
	dsgd->set_lr(0.00001);
	dsgd->set_weight_decay(0.000);


	adam_solver *randomsgd = new adam_solver(dznet);
	randomsgd->set_regularize(cacu::regularize_type::L2);
//  dsgd->load_param("/home/haofang/experiment/generative/d_solver.txt");
	randomsgd->set_beta1(0.);
	randomsgd->set_beta2(0.9);
	randomsgd->set_lr(0.00001);
	randomsgd->set_weight_decay(0.001);

	string datapath = data_path + "128_face/";
	string facepath = data_path + "128x128_celeba/";
	string trainlist = data_path + "train_list_face.txt";
	string facelist = data_path + "train_face_celeba.txt";

//	LOG_DEBUG("fuck");

	vector<string> full_data;
	vector<string> face_data;

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

	/**
	 * read train face list data into local memory
	 */
	ifstream is_face = ifstream(facelist);
	is_face.precision(numeric_limits<float>::digits10);
	if (!is_face)
		LOG_FATAL("file %s cannot be opened!", facelist.c_str());
	file_ = "";
	while (getline(is_face, file_)) {
		face_data.push_back(file_);
	}
	is_face.close();


	int ALL_DATA_SIZE = full_data.size();
	int ALL_FACE_DATA_SIZE = face_data.size();

	blob *goutput = (blob*)generator->get_oblob();
	blob *ghead = generator->layers(0)->get_head_op()->in_data<blob>();
	bin_blob *glabel = gnet->input_blobs()->asbinblob(1);

	blob *dhead = discriminator->layers(0)->get_head_op()->in_data<blob>();
	blob *doutput= (blob*)discriminator->get_oblob();
	bin_blob *dlabel = dnet->input_blobs()->asbinblob(1);

	blob *random_z = (blob*)discriminator_randomz->get_oblob();
	blob *random_head = discriminator_randomz->layers(0)->get_head_op()->in_data<blob>();

	blob *kphead = discriminator_kp->layers(0)->get_head_op()->in_data<blob>();
	blob *kpoutput = kpnet->output_blob();

	bin_blob *gtlabel = new bin_blob(batch_size, 1, 1, 1, 1, test);

	/**
	 * read data for testing
	 */

	int step_index = 0;
	int step_index_test = 0;
	int step_index_train = 0;
	int step_face_index_train = 0;

	time_utils *timer = new time_utils();

	unsigned int max_index;
	cacu::float_t count = 0;

	int previous_count=0;
	int allcount = 0;

	cacu::float_t dloss1 = 0, dloss2 = 0, gloss = 0, kploss = 0;

	blob *kpblob = new blob(batch_size, 136, 1, 1, 0, train);
	blobs *abse_datas = new blobs();
	abse_datas->push_back(kpoutput);
	abse_datas->push_back(kpblob);

	abse_loss_op *loss_abse = new abse_loss_op(abse_datas);
	loss_abse->set_loss_weight(0.5);

	bin_blob *gan_label = new bin_blob(batch_size, 1, 1, 1, 0, test);
	blobs *gan_datas = new blobs();
	gan_datas->push_back(doutput);
	gan_datas->push_back(gan_label);

	hinge_loss_op *loss_dis = new hinge_loss_op(gan_datas);
//	sigmoid_with_loss_op *loss_op = dnet->get_ops<sigmoid_with_loss_op>()[0];

	int n_s = 5;

	int n_r = 5;

	int n_b = 16;

	random_shuffle(full_data.begin(), full_data.end());
	for (int i = 1; i <= max_iter; ++i) {
		timer->start();
		for (int b = 0 ; b < n_b ; ++b){
			for (int j = 0 ; j < batch_size ; ++j)
			{
				if (step_index == ALL_DATA_SIZE) {
					step_index = 0;
					random_shuffle(full_data.begin(), full_data.end());
				}
				file_ = full_data[step_index];
				vec = split(file_, " ");
				label[0] = strtoul(vec[1].c_str(), NULL, 10);
	//			readimg(dhead->p_data(j), (datapath + vec[0]).c_str());
				imageio_utils::center_clip_imread_element_wise_norm(dhead->p_data(j),(datapath + vec[0]).c_str(), 128, 128, NULL);
				dlabel->copy2data(label, j);
				label[0] = randint(CLASS_NUM);
				glabel->copy2data(label, j);
				step_index += 1;
			}
			cacu_sdxsize(dhead->s_data(), dhead->count(), -127.5, 127.5, dhead->s_data());

			//train discriminator by true image

			dnet->set_update_weight(true);
			gan_label->set_data(1);
			//dnet->set_phase(train);
			dnet->forward_propagate();
			loss_dis->infer();
			loss_dis->derivative();
			dnet->backward_propagate();
			//dsgd->updates(i);
			dloss1 = loss_dis->loss();

			//generate the fake image
			ghead->set_init_type(uniform, 1);

			gnet->set_phase(test);
			gnet->forward_propagate();
			dhead->copy2data(goutput);
	//		cacu_copy(gnet->output_blob()->s_data(), gnet->output_blob()->count(), dhead->s_data());

			//train discriminator by fake label
			gan_label->set_data(0);
			//dnet->set_phase(train);
			dlabel->copy_blob(glabel);
			dnet->forward_propagate();
			loss_dis->infer();
			loss_dis->derivative();
			dnet->backward_propagate();
			if(b == n_b - 1)
				dsgd->updates(i);
			dloss1 += loss_dis->loss();
		}

		if(i % n_s == 0){
			for(int b = 0 ; b < n_b; ++b){
				for(int j = 0; j < batch_size; ++j){
					label[0] = randint(CLASS_NUM);
					glabel->copy2data(label, j);
				}

				//train generator by zero label
				//generate the fake image
				gnet->set_update_weight(true);
				dhead->_RESET_DIFF();
				ghead->set_init_type(uniform, 1);

				gnet->set_phase(train);
				gnet->forward_propagate();
				dhead->copy2data(goutput);

				dnet->set_update_weight(false);
				gan_label->set_data(0);
				//dnet->set_phase(train);
				dlabel->copy_blob(glabel);
				dnet->forward_propagate();
				loss_dis->infer();
				loss_dis->derivative();
				dnet->backward_propagate();
				//dsgd->updates(i);

				dloss2 = loss_dis->loss();
				goutput->copy2diff(dhead);

				gsgd->update_direction(maximize);
				gsgd->net()->backward_propagate();
				dnet->_RESET_WEIGHT_DIFF();
				if(b == n_b - 1)
					gsgd->updates(i / n_s);
			}
		}

		if(i % n_r == 0){
			for(int b = 0 ; b < 1; ++b){
				for(int j = 0; j < batch_size; ++j){
					label[0] = randint(CLASS_NUM);
					glabel->copy2data(label, j);
					if (step_index_train == ALL_DATA_SIZE) {
						step_index_train = 0;
//						random_shuffle(full_data.begin(), full_data.end());
					}
					file_ = full_data[step_index_train];
					vec = split(file_, " ");
					label[0] = strtoul(vec[1].c_str(), NULL, 10);
					imageio_utils::center_clip_imread_element_wise_norm(kphead->p_data(j),(datapath + vec[0]).c_str(), 128, 128, NULL);
					step_index_train += 1;
				}
				cacu_sdxsize(kphead->s_data(), kphead->count(), -127.5, 127.5, kphead->s_data());
				kpblob->copy2data(kpoutput);

				random_head->copy2data(kphead);

				dznet->set_update_weight(true);
				dznet->forward_propagate();

				gnet->set_update_weight(false);
				//train generator by zero label
				//generate the fake image
				ghead->_RESET_DIFF();
				ghead->copy2data(random_z);

				gnet->set_phase(train);
				gnet->forward_propagate();

				kphead->copy2data(goutput);

				kpnet->set_update_weight(false);
				kpnet->forward_propagate();
				loss_abse->infer();
				kploss = loss_abse->loss();
				loss_abse->derivative();
				kpnet->backward_propagate();

				goutput->copy2diff(kphead);

				gnet->backward_propagate();
				random_z->copy2diff(ghead);
				dznet->backward_propagate();

				if(b == n_b - 1)
					randomsgd->updates(i / n_r);

				gnet->_RESET_WEIGHT_DIFF();
				kpnet->_RESET_WEIGHT_DIFF();
			}
		}

		//*/
		timer->end();

		if (i % 1 == 0) {

			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, dsgd->lr(), timer->get_time_span() / 1000);
			LOG_INFO("discriminator loss : %f,  generator loss : %f", dloss1 / 2, dloss2);
			LOG_INFO("kp loss : %f", kploss);
			//LOG_INFO("generator loss : %f", gloss);

			logger << dloss2 << endl;
			logger.flush();

			precious_logger << dloss1 / 2 << endl;
			precious_logger.flush();

			precious_train_logger << kploss << endl;
			precious_train_logger.flush();

		}


		if (i % 4000 == 0){
			dsgd->set_lr_iter(0.1);
			gsgd->set_lr_iter(0.1);
			randomsgd->set_lr_iter(0.1);
		}

		if (i % 100 == 0) {
			ostringstream oss;

			oss << root_path << "generative/" << "test_" << i << ".jpg";
			imageio_utils::imwrite(((blob*)generator->get_oblob()),oss.str().c_str());
		}
		if (i % 1000 == 0) {
			ostringstream oss;

			oss << root_path << "models/" << "generator_" << i << ".model";
			generator->save_weights(oss.str());
			oss.str("");
			oss << root_path << "models/" << "discriminator_" << i << ".model";
			discriminator->save_weights(oss.str());
			oss.str("");

			oss << root_path << "models/" << "random_" << i << ".model";
			discriminator_randomz->save_weights(oss.str());
			oss.str("");
		}
	}

	ostringstream oss;
	oss << root_path << "models/" << "generator.model";
	//generator->save_weights(oss.str());
	oss.str("");
	oss << root_path << "models/" << "discriminator.model";
	//discriminator->save_weights(oss.str());
	logger.close();
	precious_logger.close();
	precious_train_logger.close();
	precious_constant_logger.close();

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

