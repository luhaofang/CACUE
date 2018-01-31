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


#include <sys/time.h>

#include "../../mycnn.h"

#include "../../tools/imageio_utils.h"
#include "../../tools/op_injector.h"

#include "data_proc.h"
#include "resnet_18.h"
#include "resnet_50.h"
#include "vgg_net.h"
#include "mobilenet.h"



void test_net()
{
	int batch_size = 1;

	int ALLIMAGE = 1;

	int max_iter = 1;

#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif

	//network *net = create_mobilenet_without_fc(batch_size,train);//create_res18net_without_fc(batch_size,train);//create_cifar_test_net(batch_size,test);
	network *net = create_res50net_nofc(batch_size,train);//create_mobilenet_without_fc(batch_size,train);//create_res50net_nofc(batch_size,train);
	//net->load_weights("/home/seal/4T/cacue/imagenet/final_model/vgg16.model");//"/home/seal/4T/cacue/imagenet/testnet.model");
	net->load_weights("/home/seal/4T/cacue/imagenet/res50net_380000.model");
	//net->load_weights("/home/seal/4T/cacue/imagenet/final_model/mobilenet.model");
	LOG_DEBUG("%d",net->get_op(0)->in_data<blob>()->count());
	op_injector *injector = new op_injector(net->get_op(0));

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

	struct timeval start;
	struct timeval end;
	unsigned long diff;

	for (int i = 0 ; i < max_iter; ++i)
	{
		gettimeofday(&start,NULL);
		for (int j = 0 ; j < batch_size ; ++j)
		{
			if (step_index == ALLIMAGE)
				break;
			//load image data
			//readdata(full_data[step_index],input_data->p_data(j),mean_->s_data());
			//LOG_DEBUG("%s",full_data[step_index].c_str());
			//for(int c = 0 ; c < input_data->length(); c++)
			//	LOG_DEBUG("%f",input_data->s_data()[c]);
			string filepath_ = "/home/seal/py/penguin_with_sealion.jpg";
			readdata(filepath_.c_str(),input_data->p_data(j));//,mean_->s_data());
			step_index += 1;
		}
		int length = net->output_blob()->height()*net->output_blob()->height();
		net->forward_propagate();
		//cacu_copy(net->output_blob()->s_data() ,length,net->output_blob()->s_diff());
		//cuda_setvalue<float_t>(net->output_blob()->s_diff(),length,1);
		//cacu_print(net->output_blob()->s_data(),1024);
		int max_index = argmax(net->output_blob()->s_data(),output_data->length());
		//cout << max_index << endl;
		//cacu_print(net->output_blob()->s_data(),1000);
		cuda_setvalue<float_t>(net->output_blob()->s_diff() + max_index,1,1);
		//net->output_blob()->s_diff()[0] = 1;
		//cacu_copy(net->output_blob()->s_data(),net->output_blob()->count(),net->output_blob()->s_diff());
		//for(int c = 0 ; c < net->output_blob()->channel();++c)
		//	cacu_copy(net->output_blob()->s_data() + 7 * 4 + 5 + c * length,1,net->output_blob()->s_diff() + 7 * 4 + 6 + c * length);
		net->set_weights_type(constant,0.03);
		/*
		length = net->layers(0)->get_op<convolution_op>(0)->get_weight(0)->channel_length();
		for(int o = 0 ; o < net->layers(0)->get_op<convolution_op>(0)->get_weight(0)->num(); ++o)
		{
			cuda_setvalue<float_t>(net->layers(0)->get_op<convolution_op>(0)->get_weight(0)->p_data(o) + 1 * length,1.0,length);

		}
		*/
		//net->layers(0)->get_op<convolution_op>(0)->get_weight(0)->set_init_type(msra,0.1);
		net->set_phrase(train);
		net->back_propagate();
		/*

		for(int c = 0 ; c < net->output_blob()->channel(); ++c){
			readdata(full_data[step_index],input_data->p_data(0),mean_->s_data());
			net->load_weights("/home/seal/4T/cacue/imagenet/final_model/vgg16.model");
			net->predict();
			net->output_blob()->set_diff(0);
			int length = net->output_blob()->height()*net->output_blob()->height();
			cacu_copy(net->output_blob()->s_data()+length*c,length,net->output_blob()->s_diff()+length*c);
			net->set_weights_type(constant,0.01);
			net->back_propagate();
			ostringstream oss;
			oss << "/home/seal/py/channel_feature/" << c << ".txt";
			injector->s_diff_serializa(oss.str());
		}
		//*/
		gettimeofday(&end,NULL);

		if(i % 1 == 0){
			diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
			LOG_INFO("iter_%d , %ld ms/iter", i, diff/1000);
		}
		if (step_index == ALLIMAGE)
			break;
	}

	LOG_INFO("precious: %f,%f", count / ALLIMAGE, count);
	//net->save_weights("/home/seal/4T/cacue/imagenet/final_model/vgg16_nofc.model");
	injector->s_diff_serializa("/home/seal/4T/cacue/imagenet/test.txt");

	delete injector;
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
}
