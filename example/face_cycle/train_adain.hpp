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

#ifndef TRAIN_ADAIN_FACE_CGAN_HPP_
#define TRAIN_ADAIN_FACE_CGAN_HPP_


#include "../../tools/time_utils.h"
#include "../../cacu/cacu.h"

#include "../../cacu/solvers/solvers.h"

#include "../../tools/string_utils.h"

#include "cycle_models.h"
#include "../imagenet/data_proc.h"

using namespace cacu;

void train_net() {
	int batch_size = 1;

	int max_iter = 2000000;

	int test_iter = 100;
	int train_test_iter = 100;

	string root_path = "/home/seallhf/4T/dataset/";

	string dir_path = "/home/seallhf/4T/dataset/cgan/";

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
//	network *znet = new network();

	layer_block *generator = create_generator_cycleface_adain_256(batch_size, 96, train);
	layer_block *discriminator = create_discriminator_cycleface_bn_256(batch_size, 128, train);
//	layer_block *hieratical_z = create_random_cycleface(batch_size, 64, train);


	*gnet << generator;
//	generator->load_weights(dir_path + "models/" + "generator_175000_.model");
	*dnet << discriminator;
//	discriminator->load_weights(dir_path + "models/" + "discriminator_25000.model");
//	*znet << hieratical_z;
//	hieratical_z->load_weights(dir_path + "models/" + "hieratical_z_25000.model");


	adam_solver *gsgd = new adam_solver(gnet);
	gsgd->set_regularize(cacu::regularize_type::L2);
//	gsgd->load_param("/home/haofang/experiment/generative/g_solver.txt");
	gsgd->set_beta1(0.);
	gsgd->set_beta2(0.9);
	gsgd->set_lr(0.0001);
	gsgd->set_weight_decay(0.000);

	adam_solver *dsgd = new adam_solver(dnet);
	dsgd->set_regularize(cacu::regularize_type::L2);
//  dsgd->load_param("/home/haofang/experiment/generative/d_solver.txt");
	dsgd->set_beta1(0.);
	dsgd->set_beta2(0.9);
	dsgd->set_lr(0.0001);
	dsgd->set_weight_decay(0.000);

//	adam_solver *zsgd = new adam_solver(znet);
//	zsgd->set_regularize(cacu::regularize_type::L2);
////  dsgd->load_param("/home/haofang/experiment/generative/d_solver.txt");
//	zsgd->set_beta1(0.);
//	zsgd->set_beta2(0.9);
//	zsgd->set_lr(0.0001);
//	zsgd->set_weight_decay(0.0001);

	string datapath = root_path;
	string trainlist = root_path + "HQ_facedata_list_256.txt";

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

	blob *goutput = (blob*)generator->get_oblob();
	blob *random_z_ = generator->layers(0)->get_head_op()->in_data<blob>();
	bin_blob *glabel = gnet->input_blobs()->asbinblob(1);

//	vector<inner_product_op*> ips = znet->get_ops<inner_product_op>();

//	blob *input_random = ips[ips.size() - 1]->out_data<blob>();

//	blob *h_z_0 = ips[3]->out_data<blob>();
//	blob *h_z_1 = ips[4]->out_data<blob>();
//	blob *h_z_2 = ips[5]->out_data<blob>();
//	blob *h_z_3 = ips[6]->out_data<blob>();
//	blob *zhead = znet->input_blobs()->asblob(0);

	blob *dhead = discriminator->layers(0)->get_head_op()->in_data<blob>();
	blob *doutput= (blob*)discriminator->get_oblob();
//	bin_blob *dlabel = dnet->input_blobs()->asbinblob(1);

//	bin_blob *gan_label = dnet->get_ops<sigmoid_with_loss_op>()[0]->in_datas()->asbinblob(1);

	bin_blob *gan_label = new bin_blob(batch_size, 1, 1, 1, 0, test);
	blobs *gan_datas = new blobs();
	gan_datas->push_back(doutput);
	gan_datas->push_back(gan_label);

	hinge_loss_op *loss_dis_s = new hinge_loss_op(gan_datas);

	layer *loss_ = new layer();
	loss_->op<mean_op>(doutput)->op<scalex_op>(new op_args(-1.0));

	network *loss_g = new network();
	*loss_g << loss_;

	blob *loss_g_output = loss_g->get_op<scalex_op>(1)->out_data<blob>();

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

//	sigmoid_with_loss_op *loss_dis_s = dnet->get_ops<sigmoid_with_loss_op>()[0];

	int n_s = 5;
	int n_b = 1;

//	random_z_->set_init_type(uniform, 1);

	vector<conditional_bn_op*> bns = gnet->get_ops<conditional_bn_op>();
	LOG_DEBUG("conditional_bn size : %d", bns.size());

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
				imageio_utils::center_clip_imread_element_wise_norm(dhead->p_data(j),(datapath + file_).c_str(), std_size, std_size, NULL);
				label[0] = randint(C_BN_NUM);//j;
				glabel->copy2data(label, j);
				step_index += 1;
			}
			cacu_sdxsize(dhead->s_data(), dhead->count(), -127.5, 127.5, dhead->s_data());

			//train discriminator by true image
			dnet->set_update_weight(true);
			gan_label->set_data(1);
			//dnet->set_phase(train);
			dnet->forward_propagate();
			loss_dis_s->infer();
			dloss1 = (loss_dis_s->loss()) / 2;
			loss_dis_s->derivative();
			dnet->backward_propagate();
			//dsgd->updates(i);

//			random_z_->set_init_type(uniform, 1);
//			zhead->copy2data(random_z_);
//			zhead->set_init_type(uniform, 1);
			//generate the fake image
//			znet->set_phase(test);
//			znet->forward_propagate();
//			for (int j = 0 ; j < batch_size ; ++j){
//				for(int bn_c = 0; bn_c < bns.size() / 2; ++bn_c){
//				//z0
//					blob *h_z = ips[8 + bn_c]->out_data<blob>();
////					cacu_print(h_z->s_data(), 100);
////					h_z->blob_size();
//					CHECK_EQ_OP(h_z->length() / 2, bns[bn_c * 2]->get_scales()->at(j)->count(), "%d vs %d", h_z->length() / 2, bns[bn_c * 2]->get_scales()->at(j)->count());
//					CHECK_EQ_OP(h_z->length() / 2, bns[bn_c * 2]->get_shifts()->at(j)->count(), "");
//					CHECK_EQ_OP(h_z->length() / 2, bns[bn_c * 2 + 1]->get_scales()->at(j)->count(), "");
//					CHECK_EQ_OP(h_z->length() / 2, bns[bn_c * 2 + 1]->get_shifts()->at(j)->count(), "");
//					cacu_copy(h_z->p_data(j), h_z->length() / 2, bns[bn_c * 2]->get_scales()->at(j)->s_data());
//					cacu_copy(h_z->p_data(j) + h_z->length() / 2, h_z->length() / 2, bns[bn_c * 2]->get_shifts()->at(j)->s_data());
//					cacu_copy(h_z->p_data(j), h_z->length() / 2, bns[bn_c * 2 + 1]->get_scales()->at(j)->s_data());
//					cacu_copy(h_z->p_data(j) + h_z->length() / 2, h_z->length() / 2, bns[bn_c * 2 + 1]->get_shifts()->at(j)->s_data());
//				}
//			}
//			random_z_->copy2data(input_random);
			random_z_->set_init_type(gaussian, 1);
			gnet->forward_propagate();
			dhead->copy2data(goutput);

			//train discriminator by fake label
			gan_label->set_data(0);
			//dnet->set_phase(train);
			dnet->forward_propagate();
			loss_dis_s->infer();
			dloss1 += (loss_dis_s->loss()) / 2;
			loss_dis_s->derivative();
			dnet->backward_propagate();
			if(b == n_b - 1)
				dsgd->updates(i);
		}
		if(i % n_s == 0){

			for(int b = 0; b < n_b; ++b){
				gnet->set_update_weight(true);
				//train generator by zero label
				//generate the fake image
				dhead->_RESET_DIFF();
//				loss_dis_s->set_loss_weight(1.0);
//				random_z_->set_init_type(uniform, 1);
//				zhead->copy2data(random_z_);
//				zhead->set_init_type(uniform, 1);
//				znet->set_phase(train);
//				znet->forward_propagate();
//				for (int j = 0 ; j < batch_size ; ++j){
//					for(int bn_c = 0; bn_c < bns.size() / 2; ++bn_c){
//					//z0
//						blob *h_z = ips[8 + bn_c]->out_data<blob>();
//						cacu_copy(h_z->p_data(j), h_z->length() / 2, bns[bn_c * 2]->get_scales()->at(j)->s_data());
//						cacu_copy(h_z->p_data(j) + h_z->length() / 2, h_z->length() / 2, bns[bn_c * 2]->get_shifts()->at(j)->s_data());
//						cacu_copy(h_z->p_data(j), h_z->length() / 2, bns[bn_c * 2 + 1]->get_scales()->at(j)->s_data());
//						cacu_copy(h_z->p_data(j) + h_z->length() / 2, h_z->length() / 2, bns[bn_c * 2 + 1]->get_shifts()->at(j)->s_data());
//					}
//				}
//				random_z_->copy2data(input_random);
				random_z_->set_init_type(gaussian, 1);
				gnet->forward_propagate();
				dhead->copy2data(goutput);

				dnet->set_update_weight(false);
				gan_label->set_data(0);
				//dnet->set_phase(train);

				dnet->forward_propagate();
				loss_g->forward_propagate();
				dloss2 = (loss_g_output->s_data_cpu()[0]);
				loss_g_output->set_diff(1.);
				loss_g->backward_propagate();
				dnet->backward_propagate();
				//dsgd->updates(i);
				//dloss2 = loss_gen->get_op<mean_op>(1)->mean();
				goutput->copy2diff(dhead);
				//gnet->set_phase(train);
//				gsgd->update_direction(maximize);
				gnet->backward_propagate();

//				for (int j = 0 ; j < batch_size ; ++j){
//					for(int bn_c = 0; bn_c < bns.size() / 2; ++bn_c){
//						//z0
//						blob *h_z = ips[8 + bn_c]->out_data<blob>();
//						cacu_copy(bns[bn_c * 2]->get_scales()->at(j)->s_diff(), h_z->length() / 2, h_z->p_diff(j));
//						cacu_copy(bns[bn_c * 2]->get_shifts()->at(j)->s_diff(), h_z->length() / 2, h_z->p_diff(j) + h_z->length() / 2);
//						cacu_saxpby(bns[bn_c * 2 + 1]->get_scales()->at(j)->s_diff(), 1, h_z->p_diff(j), 1, h_z->length() / 2);
//						cacu_saxpby(bns[bn_c * 2 + 1]->get_shifts()->at(j)->s_diff(), 1, h_z->p_diff(j) + h_z->length() / 2, 1, h_z->length() / 2);
//					}
//				}
//				input_random->copy2diff(random_z_);
////				cacu_print(ips[6]->out_data<blob>()->s_diff(), 100);
//				zsgd->update_direction(maximize);
//				znet->backward_propagate();

				if(b == n_b - 1){
					gsgd->updates(i / n_s);
//					zsgd->updates(i / n_s);
				}
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


		if (i % (5 * 100000 / batch_size) == 0){
			dsgd->set_lr_iter(0.1);
			gsgd->set_lr_iter(0.1);
//			zsgd->set_lr_iter(0.1);
		}

//		if (i == (3 * 100000 / batch_size)){
//			dsgd->set_lr_iter(0.1);
//			gsgd->set_lr_iter(0.1);
////			zsgd->set_lr_iter(0.1);
//		}

		if (i % 100 == 0) {
			ostringstream oss;

			oss << dir_path << "generative/" << "test_" << i << ".jpg";
			imageio_utils::imwrite(((blob*)generator->get_oblob()),oss.str().c_str());
		}
		if (i % (100000 / batch_size) == 0) {
			ostringstream oss;

			oss << dir_path << "models/" << "generator_" << i << ".model";
			generator->save_weights(oss.str());
			oss.str("");
			oss << dir_path << "models/" << "discriminator_" << i << ".model";
			discriminator->save_weights(oss.str());
//			oss.str("");
//			oss << dir_path << "models/" << "hieratical_z_" << i << ".model";
//			hieratical_z->save_weights(oss.str());
//			oss.str("");
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
