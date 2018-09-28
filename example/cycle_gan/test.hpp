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

#ifndef TEST_CYCLE_GAN_HPP_
#define TEST_CYCLE_GAN_HPP_

#include "../../cacu/cacu.h"

#include "../../cacu/solvers/solvers.h"
//#include "../../cacu/solvers/sgd_solver.h"

#include "../../tools/string_utils.h"
#include "../frcnn/tools_frcnn.h"

#include "../imagenet/data_proc.h"
#include "models.h"

using namespace cacu;

void test_net() {
	int batch_size = 1;

	int max_iter = 50000;

	int test_iter = 100;
	int train_test_iter = 100;
	int img_size = 256;

	string root_path = "/home/seallhf/4T/dataset/cycleGAN/datasets/horse2zebra/";

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
	blob *xblob_ = cacu_allocator::create_blob(batch_size, 3, img_size, img_size, train);
	blob *yblob_ = cacu_allocator::create_blob(batch_size, 3, img_size, img_size, train);

	network *xgnet = new network();
	network *ygnet = new network();


	layer_block *xgenerator = create_generator_cycleGan(batch_size, 32, train);
	xgenerator->load_weights(root_path + "models/xgenerator_50000_256.model");
	layer_block *ygenerator = create_generator_cycleGan(batch_size, 32, train);
	ygenerator->load_weights(root_path + "models/ygenerator_50000_256.model");

	/*
	 * output of the generator
	 */
	blob *xsuspicious = (blob*)xgenerator->get_oblob();
	blob *xghead = xgenerator->layers(0)->get_head_op()->in_data<blob>();
	blob *ysuspicious = (blob*)ygenerator->get_oblob();
	blob *yghead = ygenerator->layers(0)->get_head_op()->in_data<blob>();

	*xgnet << xgenerator;
	*ygnet << ygenerator;

	LOG_DEBUG("Finish creating models!");

	//string datapath = "/home/haofang/experiment/imagenet/../../data/pascalvoc/VOCdevkit/VOC2012/objects/";
	//string trainlist = "/home/haofang/experiment/imagenet/../../data/pascalvoc/VOCdevkit/VOC2012/object_list.txt";
	string xdatapath = "/home/seallhf/4T/dataset/cycleGAN/datasets/horse2zebra/testA/";
	string ydatapath = "/home/seallhf/4T/dataset/cycleGAN/datasets/horse2zebra/testB/";
	string xtrainlist = "/home/seallhf/4T/dataset/cycleGAN/datasets/horse2zebra/testA.txt";
	string ytrainlist = "/home/seallhf/4T/dataset/cycleGAN/datasets/horse2zebra/testB.txt";

	vector<string> xfull_data;
	vector<string> yfull_data;

	vector<string> vec;
	vec_i label(batch_size);

	/**
	 * read train list data into local memory
	 */
	ifstream is = ifstream(xtrainlist);
	is.precision(numeric_limits<float>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", xtrainlist.c_str());
	string file_x = "";
	while (getline(is, file_x)) {
		xfull_data.push_back(file_x);
	}
	is.close();

	//random_shuffle(xfull_data.begin(), xfull_data.end());

	/**
	 * read train list data into local memory
	 */
	is = ifstream(ytrainlist);
	is.precision(numeric_limits<float>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", ytrainlist.c_str());
	string file_y = "";
	while (getline(is, file_y)) {
		yfull_data.push_back(file_y);
	}
	is.close();

	//random_shuffle(yfull_data.begin(), yfull_data.end());

	int X_ALL_DATA_SIZE = xfull_data.size();
	int Y_ALL_DATA_SIZE = yfull_data.size();

	int step_index_x = 0;
	int step_index_y = 0;
	int step_index_test = 0;
	int step_index_train = 0;
	time_utils *timer = new time_utils();

	unsigned int max_index;
	cacu::float_t count = 0;

	int previous_count=0;
	int allcount = 0;
	cacu::float_t xdlossf = 0,xdlosst = 0,xgloss = 0;
	cacu::float_t ydlossf = 0,ydlosst = 0,ygloss = 0;
	cacu::float_t ccloss;

	for (int i = 1; i <= X_ALL_DATA_SIZE; ++i) {

		timer->start();

		/*
		 * read original img data
		 */

		if (step_index_x == X_ALL_DATA_SIZE) {
			step_index_x = 0;
			//random_shuffle(xfull_data.begin(), xfull_data.end());
		}
		if (step_index_y == Y_ALL_DATA_SIZE) {
			step_index_y = 0;
			//random_shuffle(xfull_data.begin(), xfull_data.end());
		}
		file_x = xfull_data[step_index_x];
		//load ximage data
		readimg(xblob_->s_data(),(xdatapath + file_x).c_str());

		file_y = yfull_data[step_index_y];
		//load yimage data
		readimg(yblob_->s_data(),(ydatapath + file_y).c_str());

		step_index_x += 1;
		step_index_y += 1;

		//cacu_print(yblob_->s_data(), 100);

		xghead->copy_blob(xblob_);
		xgnet->forward_propagate();

		yghead->copy_blob(yblob_);
		ygnet->forward_propagate();

		timer->end();

		ostringstream oss;
		oss.str("");
		oss << root_path << "test_result/" << file_x;
		imageio_utils::imwrite_gpu(((blob*)xgenerator->get_oblob()),oss.str().c_str());
		oss.str("");
		oss << root_path << "test_result/" << file_y;
		imageio_utils::imwrite_gpu(((blob*)ygenerator->get_oblob()),oss.str().c_str());

	}

	LOG_INFO("optimization is done!");

	vector<string>().swap(xfull_data);
	vector<string>().swap(yfull_data);

	delete xgenerator,ygenerator;
	delete xgnet, ygnet;
	delete xblob_;
	delete yblob_;
	delete timer;

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif
