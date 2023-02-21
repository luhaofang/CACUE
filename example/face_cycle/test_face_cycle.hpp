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

#ifndef TEST_FACE_CYCLE_HPP_
#define TEST_FACE_CYCLE_HPP_


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

	network *dnet = new network();
	network *gnet = new network();


	layer_block *discriminator = create_discriminator_random_cycleface(batch_size, test);
	layer_block *generator = create_generator_cycleface(batch_size, 64, test);

	*gnet << generator;
	generator->load_weights(root_path + "models/" + "generator_1000.model");

	*dnet << discriminator;
	discriminator->load_weights(root_path + "models/" + "random_1000.model");


	blob *random_z = (blob*)discriminator->get_oblob();
	blob *random_head = discriminator->layers(0)->get_head_op()->in_data<blob>();

	blob *goutput = (blob*)generator->get_oblob();
	blob *ghead = generator->layers(0)->get_head_op()->in_data<blob>();
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

//	string filename = "163_8568.bmp";
	string filename = "262_38887.jpg";
	string file_name = "0_29054.bmp 0_29109.bmp 0_29049.bmp 0_29026.bmp 0_29053.bmp 0_29058.bmp 0_29110.bmp 0_29070.bmp 0_29009.bmp 0_29091.bmp";
	vector<string> vec = split(file_name, " ");
	//random_shuffle(full_data.begin(), full_data.end());
	for(int i = 1; i < 110; ++i)
	{
		ostringstream oss;
		oss << data_path << i << ".jpg";

		timer->start();
//		imageio_utils::center_clip_imread_element_wise_norm(dhead->s_data(),(data_path + "128_face/" + filename).c_str(), 128, 128, NULL);
		imageio_utils::center_clip_imread_element_wise_norm(random_head->s_data(), oss.str().c_str(), 128, 128, NULL);
		cacu_sdxsize(random_head->s_data(), random_head->count(), -127.5, 127.5, random_head->s_data());

		label[0] = randint(CLASS_NUM);
		dnet->forward_propagate();
//		cacu_print(random_z->s_data(), random_z->count());
		glabel->copy2data(label, 0);
//		cacu_copy(random_kp->s_data(), random_kp->count(), random_z->s_data());
		ghead->copy2data(random_z);
//		random_z->set_init_type(uniform, 1);

		gnet->forward_propagate();

		oss.str("");
		oss << root_path << "test_img/" << i << "_test.jpg";
//		oss << root_path << "test_img/test_img.jpg";
		imageio_utils::imwrite(goutput, oss.str().c_str());

		oss.str("");
//
		oss << root_path << "test_img/" << i << ".jpg";
		imageio_utils::imwrite(random_head, oss.str().c_str());
//		cacu_print(dhead->s_data(), dhead->count());
		//generate the fake image
//		random_z_->set_init_type(gaussian, 1);
//		cacu_tanh(random_z_->s_data(), random_z_->count(), random_z_->s_data());
		//random_z_->copy_blob(x);

//		cacu_print(suspicious->s_data(), suspicious->count());
	}


	LOG_DEBUG("%s", filename.c_str());
	delete timer;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif
