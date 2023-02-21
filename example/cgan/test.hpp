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

#ifndef TEST_CGAN_HPP_
#define TEST_CGAN_HPP_


#include "../../tools/time_utils.h"
#include "../../cacu/cacu.h"

#include "../../cacu/solvers/solvers.h"

#include "../../tools/string_utils.h"

#include "models.h"
#include "../imagenet/data_proc.h"

using namespace cacu;

void spherical_guassian_init(blob *blob_){
	vec_t w(blob_->count());
	for (int i = 0; i < blob_->count(); ++i) {
		w[i] = spherical_unrand(0.5, 0, -0.5, 0.5);//urand_clip(-1, 1, 0.1, urand(0, 1)); //
	}
	blob_->copy2data(w);
}

void test_net() {
	int batch_size = 1;

	int max_iter = 1;

	int test_iter = 100;
	int train_test_iter = 100;

	string root_path = "/home/seallhf/cgan/";

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

	std::ofstream logger(root_path + "random_vector_recode.txt",
				ios::binary);
	logger.precision(std::numeric_limits<cacu::float_t>::digits10);

	network *gnet = new network();

	layer_block *generator = create_generator_128(batch_size, 64, train);

	*gnet << generator;
	generator->load_weights(root_path + "model_backup/" + "generator_270000_good.model");

	blob *suspicious = (blob*)generator->get_oblob();
	blob *random_z = gnet->input_blobs()->asblob(0);
	bin_blob *glabel = gnet->input_blobs()->asbinblob(1);

	blob *random_z_ = random_z->copy_create(test, 0);
	blob *x = random_z_->copy_create(test, 0);
	blob *y = random_z_->copy_create(test, 0);
	blob *mean = random_z_->copy_create(test, 0);

	vec_t random_z_mean(128);
	vector<string> vec;

	ifstream is = ifstream(root_path + "random_z_mean.txt");
	is.precision(numeric_limits<float>::digits10);
	if (!is)
		LOG_FATAL("file cannot be opened!");
	string file_ = "";
	while (getline(is, file_)) {
		vec = split(file_, ",");
		for(int i = 0 ; i < vec.size(); ++i)
			random_z_mean[i] = strtoul(vec[i].c_str(), NULL, 10);
	}
	is.close();

	mean->copy2data(random_z_mean);

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
	float_t interpolated_shifting_rate_ = 0.05;
	gnet->set_conditional_bn_morphing_interpolated_shifting_rate(interpolated_shifting_rate_);

	x->set_init_type(gaussian, 1);
//	cacu_tanh(x->s_data(), x->count(), x->s_data());
	y->set_init_type(gaussian, 1);
//	cacu_tanh(y->s_data(), y->count(), y->s_data());
//	x->init_blob_data(spherical_guassian_init);
//	y->init_blob_data(spherical_guassian_init);
//	cacu_print("y",y->s_data(),y->count());
	cacu_saxpby(x->s_data(), -1, y->s_data(), 1, y->count());
	cacu_scalex(y->s_data(), y->count(), interpolated_shifting_rate_);

	//random_shuffle(full_data.begin(), full_data.end());
	{
		timer->start();
		for (int j = 0 ; j < batch_size ; ++j)
		{
			label[0] = 0;
			glabel->copy2data(label, j);
			step_index += 1;
		}

		//generate the fake image
//		random_z_->set_init_type(gaussian, 1);
//		cacu_tanh(random_z_->s_data(), random_z_->count(), random_z_->s_data());
		//random_z_->copy_blob(x);
		cacu_copy(x->s_data(), x->count(), random_z_->s_data());
		for(int cls = 0; cls < CLASS_NUM - 1; ++cls)
		{
			glabel->set_data(cls);
			for(int step = 0; step <= all_step; ++step){
				gnet->set_conditional_bn_morphing(step, cls, cls + 1);
				cacu_saxpby(y->s_data(), 1, random_z_->s_data(), 1, random_z_->count());
				random_z->copy2data(random_z_);
				cacu_saxpby(mean->s_data(), -1, random_z->s_data(), 1, random_z->count());
				cacu_saxpby(mean->s_data(), 1, random_z->s_data(), 1, random_z->count());

				gnet->predict();
				//*/
				timer->end();

				ostringstream oss;
				oss << root_path << "face_test/" << cls << "_" << step << ".jpg";
				imageio_utils::imwrite(((blob*)generator->get_oblob()),oss.str().c_str());

				logger << cls << "_" << step << ".jpg" << " ";
				for(int i = 0 ; i < random_z_->count(); ++i)
					logger << random_z_->s_data_cpu()[i] << ",";
				logger << endl;
				logger.flush();
//				oss << root_path << "face_test/" << step << ".jpg";
			}
//			y->set_init_type(gaussian, 1);
//			cacu_tanh(y->s_data(), y->count(), y->s_data());
			y->init_blob_data(spherical_guassian_init);
			cacu_saxpby(random_z_->s_data(), -1, y->s_data(), 1, y->count());
			cacu_scalex(y->s_data(), y->count(), interpolated_shifting_rate_);
		}
	}
	logger.close();
	delete gnet;
	LOG_DEBUG("fuck1");
	delete timer;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif
