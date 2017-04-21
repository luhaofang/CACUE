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


#include <time.h>

#include "../../mycnn.h"

#include "../../tools/imageio_utils.h"

#include "alex_net.h"
#include "vgg_net.h"
#include "resnet_18.h"
#include "data_proc.h"

#include "../../tools/io/cpu_gpu_asyn.h"

#if __PARALLELTYPE__ == __CUDA__

void train_net()
{
	int batch_size = 128;

	int max_iter = 200000;

	//set gpu device if training by gpu
	cuda_set_device(1);

	//log output
	std::ofstream logger("/home/seal/4T/cacue/imagenet/res18net.log", ios::binary);
	logger.precision(std::numeric_limits<mycnn::float_t>::digits10);

	network *net = create_res18net(batch_size,train);//create_res18net(batch_size,train);//create_alexnet(batch_size,train);

	//net->load_weights("/home/seal/4T/cacue/imagenet/res18net.model");

	sgd_solver *sgd = new sgd_solver(net);

	sgd->set_lr(0.01f);
	sgd->set_weight_decay(0.0005f);

	string datapath = "/home/seal/4T/imagenet/224X224_train/";
	string trainlist = "/home/seal/4T/imagenet/train_list.txt";
	string meanfile = "/home/seal/4T/imagenet/224X224_mean.binproto";

	vector<string> full_data;
	vector<vec_i> full_label;

	/**
	 * load mean data
	 */
	//blob *mean_ = cacu_allocator::create_blob(1,3,227,227,test);
	vec_t mean_(KIMAGESIZE);
	imageio_utils::load_mean_file(&mean_[0],meanfile);
	/**
	 * read train list data into local memory
	 */
	ifstream is(trainlist);
	is.precision(numeric_limits<float>::digits10);
	if(!is)
		LOG_FATAL("file %s cannot be opened!",trainlist.c_str());
	string file_;
	while(getline(is,file_)){
		vector<string> vec = split(file_," ");
		full_data.push_back(datapath + vec[0]);
		vec_i label(1);
		label[0] = strtoul(vec[1].c_str(), NULL, 10);
		full_label.push_back(label);
	}

	/**
	 * read data for training
	 */
	blob *input_data = (blob*)net->input_blobs()->at(0);
	bin_blob *input_label = (bin_blob*)net->input_blobs()->at(1);

	asyn_initial(batch_size,KIMAGESIZE,max_iter,(&full_data),(&full_label),&mean_[0]);
	asyn_initial_threads();

	clock_t start,end;
	for (int i = 1 ; i <= max_iter; ++i)
	{
		start = clock();
		asyn_get_gpu_data(input_data->s_data(),input_label->s_data());
		sgd->train_iter();
		end = clock();
		if(i % 1 == 0){
			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i,sgd->lr(),end - start);
			((softmax_with_loss_op*)net->get_op(net->op_count()-1))->echo();
		}
		logger << ((softmax_with_loss_op*)net->get_op(net->op_count()-1))->loss() << endl;
		logger.flush();
		if(i % 50000 == 0)
			sgd->set_lr_iter(0.1f);
		if(i % 10000 == 0){
			ostringstream oss;
			oss << "/home/seal/4T/cacue/imagenet/res18net_" << i << ".model";
			net->save_weights(oss.str());
		}
	}

	logger.close();

	ostringstream oss;
	oss << "/home/seal/4T/cacue/imagenet/res18net_" << max_iter << ".model";
	net->save_weights(oss.str());
	for(int i = 0; i < full_label.size(); ++i)
	{
		vec_i().swap(full_label[i]);
	}
	vector<string>().swap(full_data);

	asyn_release();
	cuda_release();
	vec_t().swap(mean_);
}

#endif
