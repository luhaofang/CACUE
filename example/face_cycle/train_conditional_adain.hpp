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

#ifndef TRAIN_CONDITIONAL_ADAIN_CGAN_HPP_
#define TRAIN_CONDITIONAL_ADAIN_CGAN_HPP_


#include "../../tools/time_utils.h"
#include "../../cacu/cacu.h"

#include "../../cacu/solvers/solvers.h"

#include "../../tools/string_utils.h"

#include "cycle_models.h"
//#include "models.h"
#include "../imagenet/data_proc.h"

using namespace cacu;


void train_net() {
	int batch_size = 9;

	int max_iter = 1000000;

	int test_iter = 100;
	int train_test_iter = 100;

	string root_path_data = "/home/seallhf/4T/dataset/";
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
	network *hznet = new network();
	network *kpnet = new network();
	network *dznet = new network();

	layer_block *generator = create_generator_cycleface_adain(batch_size, 64, train);
	layer_block *discriminator = create_discriminator_cycleface_bn(batch_size, 64, train);
	layer_block *hieratical_z = create_random_cycleface(batch_size, 64, train);
	layer_block *discriminator_randomz = create_random_z_cycleface(batch_size, train);
	layer_block *discriminator_kp = create_discriminator_68regression_cycleface(batch_size, train);

	*gnet << generator;
	generator->load_weights(root_path + "models/" + "generator_20000.model");
	*dnet << discriminator;
	discriminator->load_weights(root_path + "models/" + "discriminator_20000.model");
	*kpnet << discriminator_kp;
	hieratical_z->load_weights(root_path + "models/" + "hieratical_z_20000.model");
	*hznet << discriminator_kp;
	discriminator_kp->load_weights(root_path + "models/" + "regressor.model");
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

	adam_solver *hzsgd = new adam_solver(hznet);
	hzsgd->set_regularize(cacu::regularize_type::L2);
//  dsgd->load_param("/home/haofang/experiment/generative/d_solver.txt");
	hzsgd->set_beta1(0.);
	hzsgd->set_beta2(0.9);
	hzsgd->set_lr(0.0001);
	hzsgd->set_weight_decay(0.0001);

	string datapath = root_path_data;
	string trainlist = root_path_data + "HQ_facedata_list_128.txt";

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


	int ALL_DATA_SIZE = full_data.size();
	int ALL_FACE_DATA_SIZE = face_data.size();

	blob *goutput = (blob*)generator->get_oblob();
	blob *ghead = generator->layers(0)->get_head_op()->in_data<blob>();
	bin_blob *glabel = gnet->input_blobs()->asbinblob(1);

	blob *dhead = discriminator->layers(0)->get_head_op()->in_data<blob>();
	blob *doutput= (blob*)discriminator->get_oblob();
//	bin_blob *dlabel = dnet->input_blobs()->asbinblob(1);

	blob *hzhead = (blob*)hieratical_z->get_oblob();

	vector<conditional_bn_op*> bns = gnet->get_ops<conditional_bn_op>();

	blob *random_z = (blob*)discriminator_randomz->get_oblob();
	blob *random_head = discriminator_randomz->layers(0)->get_head_op()->in_data<blob>();

	blob *kphead = discriminator_kp->layers(0)->get_head_op()->in_data<blob>();
	blob *kpoutput = kpnet->output_blob();

	bin_blob *gtlabel = new bin_blob(batch_size, 1, 1, 1, 1, test);

	vector<inner_product_op*> ips = hznet->get_ops<inner_product_op>();

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

	random_shuffle(full_data.begin(), full_data.end());
	for (int i = 1; i <= max_iter; ++i) {
		timer->start();
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
			label[0] = j;
			glabel->copy2data(label, j);
			step_index += 1;
		}
		cacu_sdxsize(dhead->s_data(), dhead->count(), -127.5, 127.5, dhead->s_data());
		kphead->copy2data(dhead);

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
		hzhead->copy2data(ghead);
		hznet->forward_propagate();
		for (int j = 0 ; j < batch_size ; ++j){
			for(int bn_c = 0; bn_c < bns.size() / 2; ++bn_c){
			//z0
				blob *h_z = ips[5 + bn_c]->out_data<blob>();
				cacu_copy(h_z->p_data(j), h_z->length() / 2, bns[bn_c * 2]->get_scales()->at(j)->s_data());
				cacu_copy(h_z->p_data(j) + h_z->length() / 2, h_z->length() / 2, bns[bn_c * 2]->get_shifts()->at(j)->s_data());
				cacu_copy(h_z->p_data(j), h_z->length() / 2, bns[bn_c * 2 + 1]->get_scales()->at(j)->s_data());
				cacu_copy(h_z->p_data(j) + h_z->length() / 2, h_z->length() / 2, bns[bn_c * 2 + 1]->get_shifts()->at(j)->s_data());
			}
		}
//		cacu_tanh(random_z_->s_data(), random_z_->count(), random_z_->s_data());
//		random_z->init_blob_data(spherical_guassian_init);
		gnet->set_phase(test);
		gnet->forward_propagate();
		dhead->copy2data(goutput);
//		cacu_copy(gnet->output_blob()->s_data(), gnet->output_blob()->count(), dhead->s_data());

		//train discriminator by fake label
		gan_label->set_data(0);
		//dnet->set_phase(train);
//		dlabel->copy_blob(glabel);
		dnet->forward_propagate();
		loss_dis->infer();
		loss_dis->derivative();
		dnet->backward_propagate();

		dsgd->updates(i);
		dloss1 += loss_dis->loss();

		if(i % n_s == 0){
			for(int j = 0; j < batch_size; ++j){
				label[0] = j;
				glabel->copy2data(label, j);
			}

			gnet->set_update_weight(true);
			//train generator by zero label
			//generate the fake image
			dhead->_RESET_DIFF();
			ghead->set_init_type(uniform, 1);
			hzhead->copy2data(ghead);
			hznet->forward_propagate();

			for (int j = 0 ; j < batch_size ; ++j){
				for(int bn_c = 0; bn_c < bns.size() / 2; ++bn_c){
				//z0
					blob *h_z = ips[5 + bn_c]->out_data<blob>();
					cacu_copy(h_z->p_data(j), h_z->length() / 2, bns[bn_c * 2]->get_scales()->at(j)->s_data());
					cacu_copy(h_z->p_data(j) + h_z->length() / 2, h_z->length() / 2, bns[bn_c * 2]->get_shifts()->at(j)->s_data());
					cacu_copy(h_z->p_data(j), h_z->length() / 2, bns[bn_c * 2 + 1]->get_scales()->at(j)->s_data());
					cacu_copy(h_z->p_data(j) + h_z->length() / 2, h_z->length() / 2, bns[bn_c * 2 + 1]->get_shifts()->at(j)->s_data());
				}
			}
//			cacu_tanh(random_z_->s_data(), random_z_->count(), random_z_->s_data());
//			random_z->init_blob_data(spherical_guassian_init);
			gnet->set_phase(train);
			gnet->forward_propagate();
			dhead->copy2data(goutput);
//			cacu_copy(gnet->output_blob()->s_data(), gnet->output_blob()->count(), dhead->s_data());

			dnet->set_update_weight(false);
			gan_label->set_data(0);
			//dnet->set_phase(train);
//			dlabel->copy_blob(glabel);
			dnet->forward_propagate();
			loss_dis->infer();
			loss_dis->derivative();
			dnet->backward_propagate();
			//dsgd->updates(i);
			//dloss2 = loss_gen->get_op<mean_op>(1)->mean();
			dloss2 = loss_dis->loss();
			goutput->copy2diff(dhead);
//			cacu_copy(dhead->s_diff(), gnet->output_blob()->count(), gnet->output_blob()->s_diff());
//			gnet->set_phase(train);
			gsgd->update_direction(maximize);
			gsgd->net()->backward_propagate();

			for (int j = 0 ; j < batch_size ; ++j){
				for(int bn_c = 0; bn_c < bns.size() / 2; ++bn_c){
					//z0
					blob *h_z = ips[5 + bn_c]->out_data<blob>();
					cacu_copy(bns[bn_c * 2]->get_scales()->at(j)->s_diff(), h_z->length() / 2, h_z->p_diff(j));
					cacu_copy(bns[bn_c * 2]->get_shifts()->at(j)->s_diff(), h_z->length() / 2, h_z->p_diff(j) + h_z->length() / 2);
					cacu_saxpby(bns[bn_c * 2 + 1]->get_scales()->at(j)->s_diff(), 1, h_z->p_diff(j), 1, h_z->length() / 2);
					cacu_saxpby(bns[bn_c * 2 + 1]->get_shifts()->at(j)->s_diff(), 1, h_z->p_diff(j) + h_z->length() / 2, 1, h_z->length() / 2);
				}
			}
//			cacu_print(ips[6]->out_data<blob>()->s_diff(), 100);
			hzsgd->update_direction(maximize);
			hznet->backward_propagate();

			dnet->_RESET_WEIGHT_DIFF();
			gsgd->updates(i / n_s);
		}

		if(i % n_r == 0){

			for(int j = 0; j < batch_size; ++j){
				label[0] = j;
				glabel->copy2data(label, j);
			}
			kpnet->predict();
			kpblob->copy2data(kpoutput);

			random_head->copy2data(kphead);
			dznet->forward_propagate();

			gnet->set_update_weight(false);
			//train generator by zero label
			//generate the fake image
			ghead->_RESET_DIFF();
			ghead->copy2data(random_z);

			hzhead->copy2data(random_z);
			hznet->forward_propagate();

			for (int j = 0 ; j < batch_size ; ++j){
				for(int bn_c = 0; bn_c < bns.size() / 2; ++bn_c){
				//z0
					blob *h_z = ips[5 + bn_c]->out_data<blob>();
					cacu_copy(h_z->p_data(j), h_z->length() / 2, bns[bn_c * 2]->get_scales()->at(j)->s_data());
					cacu_copy(h_z->p_data(j) + h_z->length() / 2, h_z->length() / 2, bns[bn_c * 2]->get_shifts()->at(j)->s_data());
					cacu_copy(h_z->p_data(j), h_z->length() / 2, bns[bn_c * 2 + 1]->get_scales()->at(j)->s_data());
					cacu_copy(h_z->p_data(j) + h_z->length() / 2, h_z->length() / 2, bns[bn_c * 2 + 1]->get_shifts()->at(j)->s_data());
				}
			}

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

			gsgd->update_direction(minimize);
			gnet->backward_propagate();
			for (int j = 0 ; j < batch_size ; ++j){
				for(int bn_c = 0; bn_c < bns.size() / 2; ++bn_c){
					//z0
					blob *h_z = ips[5 + bn_c]->out_data<blob>();
					cacu_copy(bns[bn_c * 2]->get_scales()->at(j)->s_diff(), h_z->length() / 2, h_z->p_diff(j));
					cacu_copy(bns[bn_c * 2]->get_shifts()->at(j)->s_diff(), h_z->length() / 2, h_z->p_diff(j) + h_z->length() / 2);
					cacu_saxpby(bns[bn_c * 2 + 1]->get_scales()->at(j)->s_diff(), 1, h_z->p_diff(j), 1, h_z->length() / 2);
					cacu_saxpby(bns[bn_c * 2 + 1]->get_shifts()->at(j)->s_diff(), 1, h_z->p_diff(j) + h_z->length() / 2, 1, h_z->length() / 2);
				}
			}
			hzsgd->update_direction(minimize);
			hznet->backward_propagate();

			random_z->copy2diff(ghead);
			cacu_saxpby(hzhead->s_diff(), 1, random_z->s_diff(), 1, random_z->count());

			dznet->backward_propagate();

			randomsgd->updates(i / n_r);

			gnet->_RESET_WEIGHT_DIFF();
			hznet->_RESET_WEIGHT_DIFF();
			kpnet->_RESET_WEIGHT_DIFF();
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


		if (i % 200000 == 0){
			dsgd->set_lr_iter(0.1);
			gsgd->set_lr_iter(0.1);
			randomsgd->set_lr_iter(0.1);
		}

		if (i % 100 == 0) {
			ostringstream oss;

			oss << root_path << "generative/" << "test_" << i << ".jpg";
			imageio_utils::imwrite(((blob*)generator->get_oblob()),oss.str().c_str());
		}
		if (i % 10000 == 0) {
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

			oss << root_path << "models/" << "hieratical_z_" << i << ".model";
			hieratical_z->save_weights(oss.str());
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

