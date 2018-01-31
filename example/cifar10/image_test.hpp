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

#include "../cifar10/cifar_quick_net.h"



void test_net()
{
	int batch_size = 1;

	int ALLIMAGE = 1;

	int max_iter = 10;

#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif

	network *net = create_cifar_quick_test_net(batch_size,train);//create_res18net_without_fc(batch_size,train);//create_cifar_test_net(batch_size,test);
	//network *net = create_res50net_nofc(batch_size,train);//create_mobilenet_without_fc(batch_size,train);//create_res50net_nofc(batch_size,train);
	//net->load_weights("/home/seal/4T/cacue/imagenet/final_model/vgg16.model");//"/home/seal/4T/cacue/imagenet/testnet.model");
	//net->load_weights("/home/seal/4T/cacue/imagenet/res50net_100000.model");


	LOG_DEBUG("%d",net->get_op(0)->in_data<blob>()->count());
	//op_injector *injector = new op_injector(net->get_op(0));
	op_injector *injector = new op_injector(net->get_op(net->op_count() - 3));
	cout << net->get_op(net->op_count() - 3)->out_data<blob>()->count() << endl;
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
		net->load_weights("/home/seal/4T/cacue/cifar10/data/cifar10_quick_test.model");
		string filepath_ = "/home/seal/py/automobile5.png";
		readdata(filepath_,input_data->s_data());//,mean_->s_data());
		step_index += 1;

		int length = net->output_blob()->height()*net->output_blob()->height();
		net->forward_propagate();
		//cacu_copy(net->output_blob()->s_data() ,length,net->output_blob()->s_diff());
		//cuda_setvalue<float_t>(net->output_blob()->s_diff(),length,1);
		//cacu_print(net->output_blob()->s_data(),1024);
		int max_index = argmax(net->output_blob()->s_data(),output_data->length());
		cout << max_index << endl;
		//cacu_print(net->output_blob()->s_data(),1000);
		//cuda_setvalue<float_t>(net->output_blob()->s_diff()+i, 1, 1);
		//net->output_blob()->s_diff()[0] = 1;
		//cacu_copy(net->output_blob()->s_data()+i ,1,net->output_blob()->s_diff()+i);
		//for(int c = 0 ; c < net->output_blob()->channel();++c)
		//	cacu_copy(net->output_blob()->s_data() + 7 * 4 + 5 + c * length,1,net->output_blob()->s_diff() + 7 * 4 + 6 + c * length);
		//net->set_weights_type(constant,0.04);
		/*
		length = net->layers(0)->get_op<convolution_op>(0)->get_weight(0)->channel_length();
		for(int o = 0 ; o < net->layers(0)->get_op<convolution_op>(0)->get_weight(0)->num(); ++o)
		{
			cuda_setvalue<float_t>(net->layers(0)->get_op<convolution_op>(0)->get_weight(0)->p_data(o) + 1 * length,1.0,length);

		}
		*/
		//net->layers(0)->get_op<convolution_op>(0)->get_weight(0)->set_init_type(msra,0.1);
		//net->set_phrase(train);
		//net->back_propagate();
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


		diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
		LOG_INFO("iter_%d , %ld ms/iter", i, diff/1000);

		ostringstream oss;
		oss << "/home/seal/4T/cacue/imagenet/test/test_" << i << ".txt";
		//injector->s_diff_serializa(oss.str());

		injector->p_data_serializa(oss.str(), i * 1024, 1024);
	}
	//net->save_weights("/home/seal/4T/cacue/imagenet/final_model/vgg16_nofc.model");


	delete injector;
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
}
