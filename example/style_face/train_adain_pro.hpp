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

#ifndef TRAIN_ADAIN_PRO_STYLE_FACE_HPP_
#define TRAIN_ADAIN_PRO_STYLE_FACE_HPP_

#include "../../tools/time_utils.h"
#include "../../cacu/cacu.h"

#include "../../cacu/solvers/solvers.h"

#include "../../tools/string_utils.h"

#include "models.h"
#include "models_backup.h"
#include "cycle_models.h"
#include "../imagenet/data_proc.h"
#include "../imagenet/resnet_50.h"

#include <math.h>

using namespace cacu;

void train_net() {
	int batch_size = 8;

	int input_dim = 64;

	int max_iter = 100000 / batch_size * 20;

	int test_iter = 100;
	int train_test_iter = 100;

	string root_path = "/home/seallhf/style_face/";

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
	std::ofstream logger(dir_path + "res50netg_.log", ios::binary);
	logger.precision(std::numeric_limits<cacu::float_t>::digits10);

	//log output
	std::ofstream precious_logger(dir_path + "res50netd_.log", ios::binary);
	precious_logger.precision(std::numeric_limits<cacu::float_t>::digits10);
	//log output
	std::ofstream precious_train_logger(
			dir_path + "res50net_train_precious_.log", ios::binary);
	precious_train_logger.precision(
			std::numeric_limits<cacu::float_t>::digits10);

	/*
	 * create generative model
	 */

	network *znet = new network();

	network *gnet = create_generator(batch_size, input_dim, 7, train);
	network *dnet = create_discriminator_linear(batch_size, input_dim, 7,
			train);
	layer_block *hieratical_z = create_mapping_function(batch_size, input_dim,
			7, train);
	*znet << hieratical_z;

//	gnet->load_weights(dir_path + "models/" + "2_generator_62500.model");
//	dnet->load_weights(dir_path + "models/" + "2_discriminator_62500.model");
//	hieratical_z->load_weights(dir_path + "models/" + "2_hieratical_z_62500.model");

	adam_solver *gsgd = new adam_solver(gnet);
	gsgd->set_regularize(cacu::regularize_type::L2);
	gsgd->load_param(dir_path + "g_solver.txt");
//	gsgd->set_beta1(0.);
//	gsgd->set_beta2(0.99);
//	gsgd->set_lr(0.0001);
//	gsgd->set_weight_decay(0.000);

	adam_solver *dsgd = new adam_solver(dnet);
	dsgd->set_regularize(cacu::regularize_type::L2);
    dsgd->load_param(dir_path + "d_solver.txt");
//	dsgd->set_beta1(0.);
//	dsgd->set_beta2(0.99);
//	dsgd->set_lr(0.0001);
//	dsgd->set_weight_decay(0.000);

	adam_solver *zsgd = new adam_solver(znet);
	zsgd->set_regularize(cacu::regularize_type::L2);
	zsgd->load_param(dir_path + "z_solver.txt");
//	zsgd->set_beta1(0.);
//	zsgd->set_beta2(0.99);
//	zsgd->set_lr(0.0001);
//	zsgd->set_weight_decay(0.000);

	string datapath = root_path;

	string datapaths[] = { root_path + "8_face_HQ/", root_path + "16_face_HQ/",
			root_path + "32_face_HQ/", root_path + "64_face_HQ/", root_path
					+ "128_face_HQ/", root_path + "256_face_HQ/" };

	string trainlist = dir_path + "HQ_facedata_list_8_png.txt";

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
		vector<string> vec = split(file_, "/");
		full_data.push_back(vec[1]);
	}
	is.close();

	int ALL_DATA_SIZE = full_data.size();

	blob *goutput = (blob*) gnet->output_blob();
	blob *random_z_ = gnet->input_blobs()->asblob(0);
	bin_blob *glabel = gnet->input_blobs()->asbinblob(1);

	vector<inner_product_op*> ips = znet->get_ops<inner_product_op>();

	blob *zhead = znet->input_blobs()->asblob(0);
	blob *dhead = dnet->input_blobs()->asblob(0);
	blob *doutput = (blob*) dnet->output_blob();

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

	cacu::float_t dloss = 0, gloss = 0;
	cacu::float_t dloss1 = 0, dloss2 = 0;

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

	int n_s = 1;
	int n_b = 1;

	ostringstream oss1;
	oss1 << dir_path << "models/" << "random_data_4*4.bin";

	blob *random_data_ = new blob(1, input_dim * 16, 4, 4, 0, test);
//	random_data_->set_init_type(gaussian, 0.01);
//	random_data_->output_bin(oss1.str().c_str());
	random_data_->input_bin(oss1.str().c_str());

	for (int i = 0; i < batch_size; ++i)
		random_z_->copy2data(random_data_->s_data(), i);

	vector<conditional_in_op*> bns = gnet->get_ops<conditional_in_op>();
	LOG_DEBUG("conditional_in size : %d", (int )bns.size());

	random_shuffle(full_data.begin(), full_data.end());

	vector<sum_elemwise_op*> Gouts = gnet->get_ops<sum_elemwise_op>("Gout");
	vector<split_op*> Dins = dnet->get_ops<split_op>("Din");

	vector<scalex_op*> gscalers = gnet->get_ops<scalex_op>();
	vector<scalex_op*> dscalers = dnet->get_ops<scalex_op>();

	vector<convolution_op*> downsamplers_in = dnet->get_ops<convolution_op>(
			"downsampler_in");
	vector<convolution_op*> upsamplers_in = gnet->get_ops<convolution_op>(
			"upsampler_in");

	vector<convolution_op*> downsamplers_out = dnet->get_ops<convolution_op>(
			"downsampler_out");
	vector<convolution_op*> upsamplers_out = gnet->get_ops<convolution_op>(
			"upsampler_out");

//	LOG_DEBUG("sampler count: %d, %d, %d, %d", downsamplers_in.size(), downsamplers_out.size(), upsamplers_in.size(), upsamplers_out.size());

	//training the bottom layer 8*8

	for (int r = 2; r < 2; ++r) {
		//change tensor flowing
		dscalers[(Dins.size() - r + 2) * 4 - 3]->set_scaler(1);
		dscalers[(Dins.size() - r + 2) * 4 - 4]->set_scaler(0);

		gscalers[(r - 2) * 4]->set_scaler(1);
		gscalers[(r - 2) * 4 + 1]->set_scaler(0);

		//use resblock
		gscalers[(r - 2) * 4 + 2]->set_scaler(1);
		gscalers[(r - 2) * 4 + 3]->set_scaler(0);

		dscalers[(Dins.size() - r + 2) * 4 - 1]->set_scaler(0);
		dscalers[(Dins.size() - r + 2) * 4 - 2]->set_scaler(1);

		//copy shortcut weights
		if (r - 2 + 1 < upsamplers_out.size())
			upsamplers_in[r - 2 + 1]->weight_copy(upsamplers_out[r - 2]);
		if (downsamplers_out.size() - 2 - r + 2 >= 0)
			downsamplers_in[downsamplers_out.size() - 2 - r + 2]->weight_copy(
					downsamplers_out[downsamplers_out.size() - 1 - r + 2]);
	}

	inner_product_op *ip0 = ips[8];
	ip0->set_weight_init_type(msra);
	ip0->get_weight(0)->set_data_from(0,
			ip0->get_weight(0)->channel(),
			1);

	for (int r = 2; r < 8; ++r) {

		datapath = datapaths[r - 2];
		LOG_INFO("data location: %s", datapath.c_str());

		int img_size = int(pow(2, r + 1));
		LOG_INFO("image_size: (%d, %d) ", img_size, img_size);

		int gindex_end = gnet->get_op_index(Gouts[r - 2]);
		int dindex_start = dnet->get_op_index(Dins[Dins.size() - r + 1]);

		LOG_DEBUG("dstart: %d / %d, gend: %d / %d", dindex_start,
				dnet->op_count(), gindex_end, gnet->op_count());
		LOG_DEBUG("%s, %s", dnet->get_op(dindex_start)->get_op_name().c_str(),
				gnet->get_op(gindex_end)->get_op_name().c_str());

		goutput = gnet->get_op(gindex_end)->out_data<blob>();
		dhead = dnet->get_op(dindex_start)->in_datas()->asblob(0);

		LOG_INFO("gscaler:");
		for (unsigned int i = 0; i < gscalers.size(); ++i)
			gscalers[i]->echo();

		LOG_INFO("dscaler:");
		for (unsigned int i = 0; i < dscalers.size(); ++i)
			dscalers[i]->echo();

		if (r < 4) {
			max_iter = 100000 / batch_size * 10;
		} else {
			max_iter = 100000 / batch_size * 20;
		}

		int start_index = 1;

		inner_product_op *ip1 = ips[8 + r - 2 + 1];

		ip1->set_weight_init_type(msra);
		ip1->get_weight(1)->set_data_from(0,
				ip1->get_weight(0)->num() / 2,
				1);

//		if(r == 2)
//			start_index = 62500;

		for (int i = start_index; i <= max_iter; ++i) {
			timer->start();
			for (int b = 0; b < n_b; ++b) {
				for (int j = 0; j < batch_size; ++j) {
					if (step_index == ALL_DATA_SIZE) {
						step_index = 0;
						random_shuffle(full_data.begin(), full_data.end());

					}
					file_ = full_data[step_index];
					imageio_utils::center_clip_imread_element_wise_norm(
							dhead->p_data(j), (datapath + file_).c_str(),
							img_size, img_size, NULL);

					label[0] = j;
					glabel->copy2data(label, j);
					step_index += 1;
				}
				cacu_sdxsize(dhead->s_data(), dhead->count(), -127.5, 127.5,
						dhead->s_data());

				//train discriminator by true image
				dnet->set_update_weight(true);
				gan_label->set_data(1);
				//dnet->set_phase(train);
				dnet->forward_propagate(dindex_start);
//				blob *test = Dins[4]->in_data<blob>();
//				cacu_print(test->s_data(), test->count());
				loss_dis_s->infer();
				dloss1 = (loss_dis_s->loss()) / 2;
				loss_dis_s->derivative();
				dnet->backward_to(dindex_start);
	//			dsgd->updates(i);

				zhead->set_init_type(gaussian, 1);

				//generate the fake image
				znet->forward_propagate();
				for (int j = 0; j < batch_size; ++j) {
					for (int bn_c = 0; bn_c < bns.size() / 2; ++bn_c) {
						//z0
						blob *h_z = ips[8 + bn_c]->out_data<blob>();
//						cacu_print(ips[8 + bn_c]->get_weight(1)->s_data(), ips[8 + bn_c]->get_weight(1)->count());
//						h_z->blob_size();
//						cacu_print(h_z->s_data(), h_z->count());
						CHECK_EQ_OP(h_z->length() / 2,
								bns[bn_c * 2]->get_scales()->at(j)->count(),
								"%d vs %d", h_z->length() / 2,
								bns[bn_c * 2]->get_scales()->at(j)->count());
						CHECK_EQ_OP(h_z->length() / 2,
								bns[bn_c * 2]->get_shifts()->at(j)->count(), "");
						CHECK_EQ_OP(h_z->length() / 2,
								bns[bn_c * 2 + 1]->get_scales()->at(j)->count(),
								"");
						CHECK_EQ_OP(h_z->length() / 2,
								bns[bn_c * 2 + 1]->get_shifts()->at(j)->count(),
								"");
						cacu_copy(h_z->p_data(j), h_z->length() / 2,
								bns[bn_c * 2]->get_scales()->at(j)->s_data());
						cacu_copy(h_z->p_data(j) + h_z->length() / 2,
								h_z->length() / 2,
								bns[bn_c * 2]->get_shifts()->at(j)->s_data());
						cacu_copy(h_z->p_data(j), h_z->length() / 2,
								bns[bn_c * 2 + 1]->get_scales()->at(j)->s_data());
						cacu_copy(h_z->p_data(j) + h_z->length() / 2,
								h_z->length() / 2,
								bns[bn_c * 2 + 1]->get_shifts()->at(j)->s_data());
					}
				}

				gnet->forward_to(gindex_end);

//				ostringstream oss2;
//				oss2 << dir_path << "generative/" << "0_test_" << i << ".jpg";
//				imageio_utils::imwrite(goutput,oss2.str().c_str());

				dhead->copy2data(goutput);

				//train discriminator  by fake label
				gan_label->set_data(0);
				dnet->forward_propagate(dindex_start);
				loss_dis_s->infer();
				dloss2 = (loss_dis_s->loss()) / 2;
				loss_dis_s->derivative();
				dnet->backward_to(dindex_start);
				if (b == n_b - 1)
					dsgd->updates(i);
				dloss = dloss1 + dloss2;
			}
			if (i % n_s == 0) {

				for (int b = 0; b < n_b; ++b) {
					//train generator by zero label
					//generate the fake image
					dhead->_RESET_DIFF();
	//				loss_dis_s->set_loss_weight(1.0);

					zhead->set_init_type(gaussian, 1);
					znet->forward_propagate();
					for (int j = 0; j < batch_size; ++j) {
						for (int bn_c = 0; bn_c < bns.size() / 2; ++bn_c) {
							//z0
							blob *h_z = ips[8 + bn_c]->out_data<blob>();
							cacu_copy(h_z->p_data(j), h_z->length() / 2,
									bns[bn_c * 2]->get_scales()->at(j)->s_data());
							cacu_copy(h_z->p_data(j) + h_z->length() / 2,
									h_z->length() / 2,
									bns[bn_c * 2]->get_shifts()->at(j)->s_data());
							cacu_copy(h_z->p_data(j), h_z->length() / 2,
									bns[bn_c * 2 + 1]->get_scales()->at(j)->s_data());
							cacu_copy(h_z->p_data(j) + h_z->length() / 2,
									h_z->length() / 2,
									bns[bn_c * 2 + 1]->get_shifts()->at(j)->s_data());
						}
					}
					gnet->forward_to(gindex_end);
					dhead->copy2data(goutput);

//					ostringstream oss2;
//					oss2 << dir_path << "generative/" << "0_test_" << i << "_.jpg";
//					imageio_utils::imwrite(goutput,oss2.str().c_str());

					dnet->set_update_weight(false);
					gan_label->set_data(0);
					//dnet->set_phase(train);

					dnet->forward_propagate(dindex_start);
					loss_g->forward_propagate();
#if __USE_DEVICE__
					gloss = (loss_g_output->s_data_cpu()[0]);
#else
					gloss = (loss_g_output->s_data()[0]);
#endif
					loss_g_output->set_diff(1.);
					loss_g->backward_propagate();
					dnet->backward_to(dindex_start);
					//dsgd->updates(i);
					//dloss2 = loss_gen->get_op<mean_op>(1)->mean();
					goutput->copy2diff(dhead);
					gnet->backward_propagate(gindex_end);

					for (int j = 0; j < batch_size; ++j) {
						for (int bn_c = 0; bn_c < bns.size() / 2; ++bn_c) {
							//z0
							blob *h_z = ips[8 + bn_c]->out_data<blob>();
							CHECK_EQ_OP(h_z->length() / 2,
									bns[bn_c * 2]->get_scales()->at(j)->count(),
									"%d vs %d", h_z->length() / 2,
									bns[bn_c * 2]->get_scales()->at(j)->count());
							CHECK_EQ_OP(h_z->length() / 2,
									bns[bn_c * 2]->get_shifts()->at(j)->count(),
									"");
							CHECK_EQ_OP(h_z->length() / 2,
									bns[bn_c * 2 + 1]->get_scales()->at(j)->count(),
									"");
							CHECK_EQ_OP(h_z->length() / 2,
									bns[bn_c * 2 + 1]->get_shifts()->at(j)->count(),
									"");
							cacu_copy(
									bns[bn_c * 2]->get_scales()->at(j)->s_diff(),
									h_z->length() / 2, h_z->p_diff(j));
							cacu_copy(
									bns[bn_c * 2]->get_shifts()->at(j)->s_diff(),
									h_z->length() / 2,
									h_z->p_diff(j) + h_z->length() / 2);
							cacu_saxpby(
									bns[bn_c * 2 + 1]->get_scales()->at(j)->s_diff(),
									1, h_z->p_diff(j), 1, h_z->length() / 2);
							cacu_saxpby(
									bns[bn_c * 2 + 1]->get_shifts()->at(j)->s_diff(),
									1, h_z->p_diff(j) + h_z->length() / 2, 1,
									h_z->length() / 2);
						}
					}
					znet->backward_propagate();

					if (b == n_b - 1) {
						gsgd->updates(i / n_s);
						zsgd->updates(i / n_s);
					}
					dnet->_RESET_WEIGHT_DIFF();
				}
			}

			//*/
			timer->end();

			if (i % 1 == 0) {

				LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, dsgd->lr(),
						timer->get_time_span() / 1000);
				LOG_INFO("discriminator loss : %f, %f, %f", dloss, dloss1,
						dloss2);
				LOG_INFO("generator loss : %f", gloss);
				//LOG_INFO("generator loss : %f", gloss);

				logger << gloss << endl;
				logger.flush();

				precious_logger << dloss << endl;
				precious_logger.flush();
			}

//			if (i % (40 * 100000 / batch_size / n_b) == 0){
//				dsgd->set_lr_iter(0.1);
//				gsgd->set_lr_iter(0.1);
//				zsgd->set_lr_iter(0.1);
//			}

//			if (i % fade_iter == 0 && r >= 2 && gscalers[(r - 2) * 4 + 2]->scaler() + fade_span < 1){
//				//LOG_DEBUG("%d,%d", gscalers.size(), dscalers.size());
//				gscalers[(r - 2) * 4 + 2]->increase_iter(fade_span);
//				gscalers[(r - 2) * 4 + 3]->increase_iter(-fade_span);
//
//				dscalers[(Dins.size() - r + 2) * 4 - 1]->increase_iter(-fade_span);
//				dscalers[(Dins.size() - r + 2) * 4 - 2]->increase_iter(fade_span);
//
//			}

			if (r >= 2 && (float_t) i / (max_iter * 1 / 2) < 1) {
				gscalers[(r - 2) * 4 + 2]->set_scaler(
						(float_t) i / (max_iter * 1 / 2));
				gscalers[(r - 2) * 4 + 3]->set_scaler(
						1. - (float_t) i / (max_iter * 1 / 2));

				dscalers[(Dins.size() - r + 2) * 4 - 1]->set_scaler(
						1. - (float_t) i / (max_iter * 1 / 2));
				dscalers[(Dins.size() - r + 2) * 4 - 2]->set_scaler(
						(float_t) i / (max_iter * 1 / 2));
			}

			if (gscalers[(r - 2) * 4 + 2]->scaler() >= 1
					|| (float_t) i / (max_iter * 1 / 2) >= 1) {
				gscalers[(r - 2) * 4 + 2]->set_scaler(1);
				gscalers[(r - 2) * 4 + 3]->set_scaler(0);

				dscalers[(Dins.size() - r + 2) * 4 - 1]->set_scaler(0);
				dscalers[(Dins.size() - r + 2) * 4 - 2]->set_scaler(1);
			}

			if (i % 100 == 0) {
				ostringstream oss;

				oss << dir_path << "generative/" << r << "_test" << i << ".jpg";
				imageio_utils::imwrite(goutput, oss.str().c_str());
			}

			if (i % (100000 / batch_size) == 0) {
				//		if (i == 1000) {
				ostringstream oss;

				oss << dir_path << "models/" << r << "_generator_" << i
						<< ".model";
				gnet->save_weights(oss.str());
				oss.str("");
				oss << dir_path << "models/" << r << "_discriminator_" << i
						<< ".model";
				dnet->save_weights(oss.str());
				oss.str("");
				oss << dir_path << "models/" << r << "_hieratical_z_" << i
						<< ".model";
				hieratical_z->save_weights(oss.str());
				//			oss.str("");
			}
		}
		//change tensor flowing
		dscalers[(Dins.size() - r + 2) * 4 - 3]->set_scaler(1);
		dscalers[(Dins.size() - r + 2) * 4 - 4]->set_scaler(0);

		gscalers[(r - 2) * 4]->set_scaler(1);
		gscalers[(r - 2) * 4 + 1]->set_scaler(0);

		//use resblock
		gscalers[(r - 2) * 4 + 2]->set_scaler(1);
		gscalers[(r - 2) * 4 + 3]->set_scaler(0);

		dscalers[(Dins.size() - r + 2) * 4 - 1]->set_scaler(0);
		dscalers[(Dins.size() - r + 2) * 4 - 2]->set_scaler(1);

		//copy shortcut weights
		if (r - 2 + 1 < upsamplers_out.size())
			upsamplers_in[r - 2 + 1]->weight_copy(upsamplers_out[r - 2]);
		if (downsamplers_out.size() - 2 - r + 2 >= 0)
			downsamplers_in[downsamplers_out.size() - 2 - r + 2]->weight_copy(
					downsamplers_out[downsamplers_out.size() - 1 - r + 2]);

		ostringstream oss;
		oss << dir_path << "models/" << r << "_generator.model";
		gnet->save_weights(oss.str());
		oss.str("");
		oss << dir_path << "models/" << r << "_discriminator.model";
		dnet->save_weights(oss.str());
		oss.str("");
		oss << dir_path << "models/" << r << "_hieratical_z.model";
		hieratical_z->save_weights(oss.str());
	}

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
