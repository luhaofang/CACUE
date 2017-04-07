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

#include "./cifar_quick_net.h"
#include "./data_proc.h"


void train_net()
{
	int batch_size = 100;

	int max_iter = 5000;

#if __PARALLELTYPE__ == __GPU__
	cuda_set_device(0);
#endif

	network *net = create_cifar_quick_net(batch_size,train);

	sgd_solver *sgd = new sgd_solver(net);

	sgd->set_lr(0.001f);

	string datapath = "/home/seal/4T/cacue/cifar10/data/";
	string meanfile = "/home/seal/4T/cacue/cifar10/data/mean.binproto";

	vector<vec_t> full_data;
	vector<vec_i> full_label;
	load_data_bymean(datapath, meanfile, full_data, full_label);

	blob *input_data = (blob*)net->input_blobs()->at(0);
	bin_blob *input_label = (bin_blob*)net->input_blobs()->at(1);

	int step_index = 0;
	clock_t start,end;
	for (int i = 0 ; i < max_iter; ++i)
	{
		start = clock();
		for (int j = 0 ; j < batch_size ; ++j)
		{
			if (step_index == kCIFARDataCount)
				step_index = 0;
			input_data->copy_data_io(full_data[step_index], j);
			input_label->copy_data_io(full_label[step_index],j);
			step_index += 1;
		}
		sgd->train_iter();
		end = clock();

		if(i % 1 == 0){
			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i,sgd->lr(),end - start);
			((softmax_with_loss_op*)net->get_op(net->op_count()-1))->echo();
		}

		if(i == 4000)
			sgd->set_lr_iter(0.1f);

	}

	net->save_weights("/home/seal/4T/cacue/cifar10/data/cifar10_quick.model");
#if __PARALLELTYPE__ == __GPU__
	cuda_release();
#endif
}
