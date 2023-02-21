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

#ifndef TEST_REGRESSOR_HPP_
#define TEST_REGRESSOR_HPP_


#include "../../tools/time_utils.h"
#include "../../cacu/cacu.h"

#include "../../cacu/solvers/solvers.h"

#include "../../tools/string_utils.h"

#include "models.h"
#include "../imagenet/data_proc.h"

using namespace cacu;


void test_net() {
	int batch_size = 1;

	int max_iter = 1;

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

	/*
	 * create generative model
	 */

	network *dnet = new network();

	layer_block *discriminator = create_discriminator_regression_cycleface(batch_size, test);

	*dnet << discriminator;
	discriminator->load_weights(root_path + "models/" + "regressor_100000.model");

	blob *suspicious = (blob*)discriminator->get_oblob();
	blob *dhead = discriminator->get_indatas()->asblob(0);


	int step_index = 0;
	int step_index_test = 0;
	int step_index_train = 0;

	time_utils *timer = new time_utils();

	unsigned int max_index;
	cacu::float_t count = 0;

	int previous_count=0;
	int allcount = 0;

	cacu::float_t dloss1 = 0,dloss2 = 0,gloss = 0;

	vec_i label(1);

	int all_step = 20;

	string filename = "163_8568.bmp";
//	string filename = "262_38887.jpg";
	//random_shuffle(full_data.begin(), full_data.end());

	vector<convolution_op *> convs = dnet->get_ops<convolution_op>();
	vector<inner_product_op *> ips = dnet->get_ops<inner_product_op>();

	{
		timer->start();
		imageio_utils::center_clip_imread_element_wise_norm(dhead->s_data(),(data_path + "128_face/" + filename).c_str(), 128, 128, NULL);
		cacu_sdxsize(dhead->s_data(), dhead->count(), -127.5, 127.5, dhead->s_data());

//		cacu_print(dhead->s_data(), dhead->count());
		//generate the fake image
//		random_z_->set_init_type(gaussian, 1);
//		cacu_tanh(random_z_->s_data(), random_z_->count(), random_z_->s_data());
		//random_z_->copy_blob(x);
		dnet->predict();
		cacu_print(suspicious->s_data(), suspicious->count());
	}

	filename = "262_38887.jpg";
	//random_shuffle(full_data.begin(), full_data.end());

//	vector<convolution_op *> convs = dnet->get_ops<convolution_op>();
//	vector<inner_product_op *> ips = dnet->get_ops<inner_product_op>();

	{
		timer->start();
		imageio_utils::center_clip_imread_element_wise_norm(dhead->s_data(),(data_path + "128_face/" + filename).c_str(), 128, 128, NULL);
		cacu_sdxsize(dhead->s_data(), dhead->count(), -127.5, 127.5, dhead->s_data());

//		cacu_print(dhead->s_data(), dhead->count());
		//generate the fake image
//		random_z_->set_init_type(gaussian, 1);
//		cacu_tanh(random_z_->s_data(), random_z_->count(), random_z_->s_data());
		//random_z_->copy_blob(x);
		dnet->predict();
		cacu_print(suspicious->s_data(), suspicious->count());
	}

//	for(int i = 0 ; i < convs.size(); ++i){
//		LOG_DEBUG("conv:");
//		cacu_print(convs[i]->in_datas()->asblob(0)->s_data(), convs[i]->in_datas()->asblob(0)->count());
//	}
//
//	for(int i = 0 ; i < ips.size(); ++i){
//		cacu_print("ip", ips[i]->in_datas()->asblob(0)->s_data(), ips[i]->in_datas()->asblob(0)->count());
//	}


	LOG_DEBUG("%s", filename.c_str());
	delete timer;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif
