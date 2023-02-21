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

#ifndef TRAIN_FACE_REGRESSOR_HPP_
#define TRAIN_FACE_REGRESSOR_HPP_


#include "../../tools/time_utils.h"
#include "../../cacu/cacu.h"

#include "../../cacu/solvers/solvers.h"

#include "../../tools/string_utils.h"

#include "models.h"
#include "../imagenet/data_proc.h"

using namespace cacu;


void train_net() {
	int batch_size = 256;

	int max_iter = 100000;

	string data_path = "/home/seallhf/cgan/";
	string root_path = "/home/seallhf/face_transform/";

	//set gpu device if training by gpu
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif
#endif
	//set random seed
	set_rand_seed();

	//log output
	std::ofstream logger(root_path + "face_regressor.log",
			ios::binary);
	logger.precision(std::numeric_limits<cacu::float_t>::digits10);

	/*
	 * create generative model
	 */

	network *dnet = new network();

	layer_block *discriminator = create_discriminator_68regression_cycleface(batch_size, train);

	*dnet << discriminator;
//	discriminator->load_weights(root_path + "models/" + "discriminator_10000.model");

	sgd_solver *dsgd = new sgd_solver(dnet);
	dsgd->set_regularize(cacu::regularize_type::L2);
//  dsgd->load_param("/home/haofang/experiment/generative/d_solver.txt");
	dsgd->set_lr(0.01);
	dsgd->set_weight_decay(0.001);

	string datapath = data_path + "128_face/";
	string trainlist = root_path + "regression68_data.txt";

	vector<string> full_data;

	vector<string> vec;
	vec_t offset(136);

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

	blob *suspicious = (blob*)discriminator->get_oblob();
	blob *random_z_ = new blob(batch_size, 136, 1, 1 ,0, train);

	blob *dhead = discriminator->get_indatas()->asblob(0);


	int step_index = 0;
	int step_index_test = 0;
	int step_index_train = 0;

	time_utils *timer = new time_utils();

	unsigned int max_index;
	cacu::float_t count = 0;

	int previous_count=0;
	int allcount = 0;

	cacu::float_t dloss1 = 0, dloss2 = 0, gloss = 0;

	blobs *gan_datas = new blobs();
	gan_datas->push_back(suspicious);
	gan_datas->push_back(random_z_);
//	LOG_DEBUG("fuck! %d", gan_datas->size());

	abse_loss_op *loss_dis = new abse_loss_op(gan_datas);

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
//			readimg(dhead->p_data(j), (datapath + vec[0]).c_str());
			imageio_utils::center_clip_imread_element_wise_norm(dhead->p_data(j),(datapath + vec[0]).c_str(), 128, 128, NULL);
			vec = split(vec[1], ",");
			for(int n = 0; n < vec.size(); ++n){
				offset[n] = strtof32(vec[n].c_str(), NULL);
			}
			random_z_->copy2data(&offset[0], j);
			step_index += 1;
		}

		cacu_sdxsize(dhead->s_data(), dhead->count(), -127.5, 127.5, dhead->s_data());
		//dnet->set_phase(train);
		dnet->forward_propagate();
		loss_dis->infer();

		loss_dis->derivative();
		dnet->backward_propagate();
		dsgd->updates(i);
		dloss1 = loss_dis->loss();

		//*/
		timer->end();


		if (i % 10 == 0) {

			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, dsgd->lr(), timer->get_time_span() / 1000);
			LOG_INFO("discriminator loss : %f", dloss1);
			//LOG_INFO("generator loss : %f", gloss);

			logger << dloss1 << endl;
			logger.flush();

		}


		if (i % 40000 == 0){
			dsgd->set_lr_iter(0.1);
		}


		if (i % 10000 == 0) {
			ostringstream oss;

			oss << root_path << "models/" << "regressor68_" << i << ".model";
			discriminator->save_weights(oss.str());
			oss.str("");
		}
	}

	ostringstream oss;
	oss << root_path << "models/" << "regressor68.model";
	discriminator->save_weights(oss.str());
	logger.close();

	LOG_INFO("optimization is done!");

	vector<string>().swap(full_data);

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
