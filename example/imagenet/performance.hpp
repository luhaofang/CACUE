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

#ifndef PERFORMANCE_HPP_
#define PERFORMANCE_HPP_

#include <time.h>

#include "../../cacu/cacu.h"

#include "../../tools/imageio_utils.h"

#include "data_proc.h"
#include "resnet_18.h"
#include "resnet_50.h"
#include "vgg_net.h"
#include "mobilenet.h"
#include "mnasnet.h"


void test_net()
{
	int batch_size = 2;

	int ALLIMAGE = 1;

	int max_iter = 10;

#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif

	network *net = create_vgg_16_net(batch_size,test);//create_vgg_16_net(batch_size,test);//create_res50net(batch_size,test);//create_cifar_test_net(batch_size,test);
	//network *net = create_vgg_16_net(batch_size, test);
	//network *net = create_mobilenet(batch_size,test);
	//net->load_weights("/home/seal/4T/cacue/imagenet/res50net_120000.model");//vggnet_40000.model");
//	net->load_weights("./mobile_net_test.model");
	//net->check();
	//op_injector *injector = new op_injector(net->get_op(29));
	//net->save_weights("./mobile_net_test.model");
	/**
	 * read data for testing
	 */
	blob *input_data = (blob*)net->input_blobs()->at(0);
	blob *output_data = net->output_blob();

	unsigned int max_index;
	cacu::float_t count = 0;

	int step_index = 0;

	time_utils *timer = new time_utils();

	for(int i = 0; i < 1; ++i){
		input_data->set_init_type(constant,0.1);
		timer->start();
		net->predict();
		timer->end();

		LOG_INFO("one iter cost, %ld ms/iter", timer->get_time_span() / 1000 );
	}

	//injector->o_blob_serializa("/home/seal/4T/cacue/imagenet/relu.txt");

	//delete injector;
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
}


#endif
