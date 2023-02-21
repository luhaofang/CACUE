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


#ifndef TEST_GAN_HPP_
#define TEST_GAN_HPP_

#include "../../cacu/cacu.h"

#include "../../tools/imageio_utils.h"
#include "../../tools/time_utils.h"

#include "generate_models.h"

using namespace cacu;
using namespace cacu_tools;

void test_net()
{
	int batch_size = 64;

	int max_iter = 1;

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif
#endif

	string datapath = "/home/seallhf/4T/dataset/celeba/data/";

	blob *random_z_ = cacu_allocator::create_blob(batch_size, 100, 1, 1, train);

	blobs *gin_blobs = new blobs();
	gin_blobs->push_back(random_z_);

	network *gnet = new network(gin_blobs);
//

	layer_block *generator = create_generator_64(random_z_, 128, test);
//	net->set_is_use_bias(true);
	generator->load_weights("/home/seallhf/4T/dataset/celeba/data/models/generator_50000_64_sn.model");
	*gnet << generator;

	time_utils *timer= new time_utils();

	for (int i = 0; i < max_iter; ++i)
	{
		timer->start();

		random_z_->set_init_type(uniform, 1);

		gnet->predict();
		ostringstream oss;
		oss << datapath << "generative_64/" << "test_" << i << "_64.jpg";
		imageio_utils::imwrite(((blob*) generator->get_oblob()), oss.str().c_str());
	}
//	imageio_utils::imwrite_channel(net->get_op(0)->out_data<blob>(), (datapath + "cifar10_quick_test.jpg").c_str());

	delete gnet;
	delete timer;
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif
