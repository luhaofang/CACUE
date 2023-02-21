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

#ifndef TEST_ADAIN_HPP_
#define TEST_ADAIN_HPP_


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

	int input_dim = 64;

	int test_iter = 100;
	int train_test_iter = 100;

	string root_path = "/home/seallhf/4T/dataset/cgan/";
	string data_path = "/home/seallhf/test_face/";

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

	network *znet = new network();


	layer_block *discriminator = create_mapping_function(batch_size, input_dim, 7, test);
	network *gnet = create_generator(batch_size, input_dim, 7, test);

	gnet->load_weights(root_path + "models/" + "generator_260000.model");

	*znet << discriminator;
	discriminator->load_weights(root_path + "models/" + "hieratical_z_260000.model");


	blob *random_z_ = (blob*)discriminator->get_oblob();
	blob *zhead = discriminator->layers(0)->get_head_op()->in_data<blob>();

	blob *goutput = (blob*)gnet->output_blob();
	blob *ghead = gnet->input_blobs()->asblob(0);
	bin_blob *glabel = gnet->input_blobs()->asbinblob(1);


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

	ostringstream oss1;
	oss1 << root_path << "models/" << "random_data_.bin";

	blob *random_data_ = new blob(1, random_size, 1, 1, 0, test);
//	random_data_->set_init_type(gaussian, 0.1);
//	random_data_->output_bin(oss1.str().c_str());
	random_data_->input_bin(oss1.str().c_str());
	for(int i = 0; i < batch_size; ++i)
		random_z_->copy2data(random_data_->s_data(), i);

	vector<inner_product_op*> ips = znet->get_ops<inner_product_op>();
	vector<conditional_in_op*> bns = gnet->get_ops<conditional_in_op>();
	zhead->set_init_type(uniform, 1);
	//generate the fake image
//	znet->set_phase(test);
	znet->forward_propagate();
	for (int j = 0 ; j < batch_size ; ++j){
		for(int bn_c = 0; bn_c < bns.size() / 2; ++bn_c){
		//z0
			blob *h_z = ips[8 + bn_c]->out_data<blob>();
//					cacu_print(h_z->s_data(), h_z->length());
//					h_z->blob_size();
			CHECK_EQ_OP(h_z->length() / 2, bns[bn_c * 2]->get_scales()->at(j)->count(), "%d vs %d", h_z->length() / 2, bns[bn_c * 2]->get_scales()->at(j)->count());
			CHECK_EQ_OP(h_z->length() / 2, bns[bn_c * 2]->get_shifts()->at(j)->count(), "");
			CHECK_EQ_OP(h_z->length() / 2, bns[bn_c * 2 + 1]->get_scales()->at(j)->count(), "");
			CHECK_EQ_OP(h_z->length() / 2, bns[bn_c * 2 + 1]->get_shifts()->at(j)->count(), "");
			cacu_copy(h_z->p_data(j), h_z->length() / 2, bns[bn_c * 2]->get_scales()->at(j)->s_data());
			cacu_copy(h_z->p_data(j) + h_z->length() / 2, h_z->length() / 2, bns[bn_c * 2]->get_shifts()->at(j)->s_data());
			cacu_copy(h_z->p_data(j), h_z->length() / 2, bns[bn_c * 2 + 1]->get_scales()->at(j)->s_data());
			cacu_copy(h_z->p_data(j) + h_z->length() / 2, h_z->length() / 2, bns[bn_c * 2 + 1]->get_shifts()->at(j)->s_data());
		}
	}
	glabel->set_data(0);
	gnet->forward_propagate();

	ostringstream oss;

	oss << root_path << "test/" << "test.jpg";
	imageio_utils::imwrite(((blob*)gnet->output_blob()),oss.str().c_str());

	delete timer;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif
