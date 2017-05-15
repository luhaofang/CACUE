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
#include "../../tools/op_injector.h"

#include "data_proc.h"
#include "resnet_18.h"
#include "vgg_net.h"



void test_net()
{
	int batch_size = 100;

	int ALLIMAGE = 50000;

	int max_iter = 500;

#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif

	network *net = create_vgg_16_net(batch_size,test);//create_vgg_16_net(batch_size,test);//create_cifar_test_net(batch_size,test);

	net->load_weights("/home/seal/4T/cacue/imagenet/vgg16net_30000.model");

	op_injector *injector = new op_injector(net->get_op(29));

	string datapath = "/home/seal/4T/imagenet/224X224_val/";
	string vallist = "/home/seal/4T/imagenet/val_list";
	string meanfile = "/home/seal/4T/imagenet/224X224_mean.binproto";

	vector<string> full_data;
	vector<unsigned int> full_label;

	/**
	 * load mean data
	 */
	blob *mean_ = cacu_allocator::create_blob(1,3,224,224,test);
	#if __PARALLELTYPE__ == __CUDA__
		imageio_utils::load_mean_file_gpu(mean_->s_data(),meanfile);
	#else
		imageio_utils::load_mean_file(mean_->s_data(),meanfile);
	#endif

	/**
	 * read test list data into local memory
	 */
	ifstream is(vallist);
	is.precision(numeric_limits<float>::digits10);
	if(!is)
		LOG_FATAL("file %s cannot be opened!",vallist.c_str());
	string file_;
	while(getline(is,file_)){
		vector<string> vec = split(file_," ");
		full_data.push_back(datapath + vec[0]);
		unsigned int label = strtoul(vec[1].c_str(), NULL, 10);
		full_label.push_back(label);
	}

	/**
	 * read data for testing
	 */
	blob *input_data = (blob*)net->input_blobs()->at(0);
	blob *output_data = net->output_blob();

	unsigned int max_index;
	float_t count = 0;

	int step_index = 0;

	clock_t start,end;

	for (int i = 0 ; i < max_iter; ++i)
	{
		start = clock();

		for (int j = 0 ; j < batch_size ; ++j)
		{
			if (step_index == ALLIMAGE)
				break;
			//load image data
			readdata(full_data[step_index],input_data->p_data(j),mean_->s_data());
			//readdata("/home/seal/4T/imagenet/test1.JPEG",input_data->p_data(j));//,mean_->s_data());
			step_index += 1;
		}

		net->predict();
		//injector->get_outblob_count();
		for(int j = 0 ; j < batch_size ; ++j)
		{
			max_index = argmax(output_data->p_data(j),output_data->length());
			cout << max_index << ":" << full_label[i * batch_size + j] << endl;
			if(max_index == full_label[i * batch_size + j]){
				count += 1.0;
			}
		}

		end = clock();

		if(i % 1 == 0){
			LOG_INFO("iter_%d , %ld ms/iter", i, end - start);
		}
		if (step_index == ALLIMAGE)
			break;
	}

	LOG_INFO("precious: %f,%f", count / ALLIMAGE, count);

	//injector->o_blob_serializa("/home/seal/4T/cacue/imagenet/relu.txt");

	delete injector;
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
}
