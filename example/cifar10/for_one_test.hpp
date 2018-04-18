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


#ifndef FOR_ONE_TEST_HPP_
#define FOR_ONE_TEST_HPP_

#include "../../cacu/cacu.h"

#include "../../tools/imageio_utils.h"
#include "../../tools/time_utils.h"
#include "../../tools/injector.h"
#include "../../tools/serializer_utils.h"

#include "cifar_test_net.h"
#include "data_proc.h"

using namespace cacu;
using namespace cacu_tools;

void test_net()
{
	int batch_size = 1;

	int max_iter = 10;

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(2);
#endif
#endif

	network *net = create_cifar_test_net(batch_size, test);

	string datapath = "/home/haofang/data/cifar10/";
	string meanfile = "/home/haofang/data/cifar10/mean.binproto";

	vector<vec_t> full_data;
	vector<vec_i> full_label;
	load_test_data(datapath, full_data, full_label);
	//load_test_data_bymean(datapath, meanfile, full_data, full_label);

	vec_i _full_label;
	for (int i = 0; i < full_label.size(); ++i)
		_full_label.push_back(full_label[i][0]);

	blob *input_data = (blob*)net->input_blobs()->at(0);

	blob *output_data = net->output_blob();
	
	injector *injector_v = new injector(net->get_op(11));
	injector *injector_s = new injector(net->get_op(12));


	net->load_weights("C:/Users/Haofang.Lu/Desktop/git/cacue_vs/example/cifar10/model_1000.model");

	unsigned int max_index;
	cacu::float_t count = 0;
	cacu_tools::timeval start;
	cacu_tools::timeval end;
	unsigned long diff;

	int step_index = 0;

	int allcount = 0;
	for (int i = 0; i < max_iter; ++i)
	{
		gettime(&start);

		input_data->resize(batch_size, 3, 32, 32);
		for (int j = 0; j < batch_size; ++j)
		{
			if (step_index == kCIFARBatchSize)
				break;
			input_data->copy2data(full_data[step_index], j);
			step_index += 1;
		}

		net->predict();
		for (int j = 0; j < batch_size; ++j)
		{
			max_index = argmax(output_data->p_data(j), output_data->length());
			if (max_index == _full_label[allcount + j]) {
				count += 1.0;
			}
		}
		allcount += batch_size;
		batch_size = 1;//urandint(10, 100);
		LOG_DEBUG("batch_size: %d", batch_size);
		gettime(&end);

		if (i % 1 == 0) {
			diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec
				- start.tv_usec;
			LOG_INFO("iter_%d, %ld ms/iter", i, diff / 1000);
		}
		if (step_index == kCIFARBatchSize)
			break;

		ostringstream oss;
		oss << "/home/haofang/experiments/cifar10/raw" << i << ".data";
		serializer::blob_serialize(net->get_op(0)->in_data<blob>(), oss.str());
		ostringstream ossv;
		ossv << "/home/haofang/experiments/cifar10/visualized" << i << ".data";
		injector_v->s_data_serializa(ossv.str());
		ostringstream osss;
		osss << "/home/haofang/experiments/cifar10/semantic" << i << ".data";
		injector_s->s_data_serializa(osss.str());

	}	


	LOG_INFO("precious: %f,%f", count / allcount, count);
	delete net;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif

