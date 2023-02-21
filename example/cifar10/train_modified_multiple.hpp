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


#ifndef TRAIN_MODIFIED_MUL_HPP_
#define TRAIN_MODIFIED_MUL_HPP_

#include <time.h>

#include "../../cacu/solvers/sgd_solver.h"
#include "../../cacu/solvers/adam_solver.h"

#include "../../cacu/cacu.h"
#include "../../cacu/config.h"

#include "../../tools/imageio_utils.h"
#include "../../tools/time_utils.h"

//#include "cifar_quick_net.h"
#include "cifar_test_net.h"
#include "data_proc.h"

using namespace cacu;
using namespace cacu_tools;

void train_net()
{
	int batch_size = 100;

	int max_iter = 5000;

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);

#endif
#endif
	//set random seed
	set_rand_seed();

	network *net = create_cifar_quick_net(batch_size,train);
	//net->load_weights_from("/home/haofang/experiment/cifar10/cifar10_quick_normal.model",10);
	//net->load_weights("C:/Users/Haofang.Lu/Desktop/git/cacue_vs/example/cifar10/model_3000.model");

	blob *temp_g1 = new blob(batch_size, 32, 16, 16, 0, test);
	blob *temp_g2 = new blob(batch_size, 32, 8, 8, 0, test);
	blob *temp_g3 = new blob(batch_size, 64, 4, 4, 0, test);

	blob *temp_d1 = new blob(batch_size, 3, 32, 32, 0, test);
	blob *temp_d2 = new blob(batch_size, 32, 16, 16, 0, test);
	blob *temp_d3 = new blob(batch_size, 32, 8, 8, 0, test);


	sgd_solver *sgd = new sgd_solver(net);
	sgd->set_lr(0.001f);
	sgd->set_momentum(0.9f);
	sgd->set_weight_decay(0.004f);
	//sgd->set_regularize(regularize_type::L1);

	string rootpath = "/Users/seallhf/Documents/datasets/cifar10/";
//	net->load_weights(rootpath + "cifar10_quick_test.model");

	std::ofstream logger(rootpath + "loss.txt", ios::binary);
	logger.precision(std::numeric_limits<cacu::float_t>::digits10);

	string meanfile = rootpath + "mean.binproto";

	vector<vec_t> full_data;
	vector<vec_i> full_label;

	load_data_bymean(rootpath, meanfile, full_data, full_label);
	//load_data(datapath, full_data, full_label);

	blob *input_data = (blob*)net->input_blobs()->at(0);
	bin_blob *input_label = (bin_blob*)net->input_blobs()->at(1);

	vector<split_op*> splits = net->get_ops<split_op>();
	vector<convolution_op*> convs = net->get_ops<convolution_op>();

	network *deconv1 = deconv(dynamic_cast<split_op*>(splits[0])->out_datas()->at(1), 3, 5, 2, 0);
	deconv1->get_op<conv_transpose_op>(0, CACU_CONV_TRANS)->set_weight_init_type(xavier);
	network *deconv2 = deconv(dynamic_cast<split_op*>(splits[1])->out_datas()->at(1), 32, 5, 2, 0);
	deconv2->get_op<conv_transpose_op>(0, CACU_CONV_TRANS)->set_weight_init_type(xavier);
	network *deconv3 = deconv(dynamic_cast<split_op*>(splits[2])->out_datas()->at(1), 32, 5, 2, 0);
	deconv3->get_op<conv_transpose_op>(0, CACU_CONV_TRANS)->set_weight_init_type(xavier);

//	deconv1->load_weights(rootpath + "deconv1.model");
//	deconv2->load_weights(rootpath + "deconv2.model");
//	deconv3->load_weights(rootpath + "deconv3.model");

	float_t deconv_lr = 0.001f;

	sgd_solver *sgd1 = new sgd_solver(deconv1);
	sgd1->set_lr(deconv_lr);
	sgd1->set_momentum(0.9f);
	sgd1->set_weight_decay(0.004f);
	sgd_solver *sgd2 = new sgd_solver(deconv2);
	sgd2->set_lr(deconv_lr);
	sgd2->set_momentum(0.9f);
	sgd2->set_weight_decay(0.004f);
	sgd_solver *sgd3 = new sgd_solver(deconv3);
	sgd3->set_lr(deconv_lr);
	sgd3->set_momentum(0.9f);
	sgd3->set_weight_decay(0.004f);

	float_t deconv_add1 = 1;
	float_t deconv_add2 = 1;
	float_t deconv_add3 = 1;

	blobs *datas1 = new blobs();
	datas1->push_back(convs[0]->in_data<blob_base>());
	datas1->push_back(deconv1->output_blob());

	layer *element_sum1 = new layer();
	element_sum1->op<add_op>(datas1);

	blobs *datas2 = new blobs();
	datas2->push_back(convs[1]->in_data<blob_base>());
	datas2->push_back(deconv2->output_blob());

	layer *element_sum2 = new layer();
	element_sum2->op<add_op>(datas2);

	blobs *datas3 = new blobs();
	datas3->push_back(convs[2]->in_data<blob_base>());
	datas3->push_back(deconv3->output_blob());

	layer *element_sum3 = new layer();
	element_sum3->op<add_op>(datas3);

	int step_index = 0;
	time_utils *timer = new time_utils();
	unsigned long diff;

	float_t predict_loss = 0.0;
	for (int i = 1 ; i < max_iter; ++i)
	{
		timer->start();
		//batch_size = 200;
		//input_data->resize(batch_size,3,32,32);
		//input_label->resize(batch_size,1,1,1);
		for (int j = 0 ; j < batch_size ; ++j)
		{
			if (step_index == kCIFARDataCount)
				step_index = 0;
			input_data->copy2data(full_data[step_index], j);
			input_label->copy2data(full_label[step_index],j);
			step_index += 1;
		}

		sgd->train_iter(i);
		//net->predict();
		predict_loss = ((softmax_with_loss_op*)net->get_op(net->op_count() - 1))->loss();

		//cacu_print(deconv3->input_blobs()->asblob(0)->s_data(), 10);

		if(i % 100 == 0){
			ostringstream oss;
			oss << rootpath + "./test_img/input_data_" << i <<".jpg";
			imageio_utils::imwrite(input_data, oss.str().c_str());
		}

		cacu_print("input", input_data->s_data(),10);
		deconv1->forward_propagate();
		cacu_copy(deconv1->input_blobs()->asblob(0)->s_data(), temp_g1->count(), temp_g1->s_data());
		cacu_scalex(deconv1->output_blob()->s_data(), deconv1->output_blob()->count(), deconv_add1);

		net->set_phase(train);
		if(i % 100 == 0){
			ostringstream oss;
			oss << rootpath + "./test_img/deconv1_" << i <<".jpg";
			imageio_utils::imwrite(deconv1->output_blob(), oss.str().c_str());
		}

//		cacu_print(convs[0]->in_data<blob>()->s_data(), 10);
//		cacu_print(convs[0]->in_data<blob>()->s_data(), 10);
		cacu_print("deconv", deconv1->output_blob()->s_data(), 10);
		element_sum1->operate();

		cacu_copy((convs[0])->in_data<blob>()->s_data(), temp_d1->count(), temp_d1->s_data());
		cacu_copy(((blob*)element_sum1->get_oblob())->s_data(), temp_d1->count(), (convs[0])->in_data<blob>()->s_data());
		if(i % 100 == 0){
			ostringstream oss;
			oss << rootpath + "./test_img/added_" << i <<".jpg";
			imageio_utils::imwrite((blob *)element_sum1->get_oblob(), oss.str().c_str());
		}
		cacu_print("added", ((blob *)element_sum1->get_oblob())->s_data(), 10);

		net->layers(0)->operate();
		net->layers(1)->operate();
		net->layers(2)->operate();

		deconv2->forward_propagate();
		cacu_copy(deconv2->input_blobs()->asblob(0)->s_data(), temp_g2->count(), temp_g2->s_data());
		cacu_scalex(deconv2->output_blob()->s_data(), deconv2->output_blob()->count(), deconv_add2);

		element_sum2->operate();
		cacu_copy((convs[1])->in_data<blob>()->s_data(), temp_d2->count(), temp_d2->s_data());
		cacu_copy(((blob*)element_sum2->get_oblob())->s_data(), temp_d2->count(), (convs[1])->in_data<blob>()->s_data());

		net->layers(3)->operate();
		net->layers(4)->operate();
		net->layers(5)->operate();

		deconv3->forward_propagate();
		cacu_copy(deconv3->input_blobs()->asblob(0)->s_data(), temp_g3->count(), temp_g3->s_data());
		cacu_scalex(deconv3->output_blob()->s_data(), deconv3->output_blob()->count(), deconv_add3);

		element_sum3->operate();
		cacu_copy((convs[2])->in_data<blob>()->s_data(), temp_d3->count(), temp_d3->s_data());
		cacu_copy(((blob*)element_sum3->get_oblob())->s_data(), temp_d3->count(), (convs[2])->in_data<blob>()->s_data());

		net->forward_propagate(6);

		net->backward_propagate(6);

		cacu_copy((convs[2])->in_data<blob>()->s_diff(), temp_d3->count(), ((blob*)element_sum3->get_oblob())->s_diff());
		//cacu_copy(temp_d3->s_data(), temp_d3->count(), (convs[2])->in_data<blob>()->s_data());
		element_sum3->grad();

		net->layers(5)->grad();
		net->layers(4)->grad();
		net->layers(3)->grad();

		cacu_copy((convs[1])->in_data<blob>()->s_diff(), temp_d2->count(), ((blob*)element_sum2->get_oblob())->s_diff());
		//cacu_copy(temp_d2->s_data(), temp_d2->count(), (convs[1])->in_data<blob>()->s_data());
		element_sum2->grad();

		net->layers(2)->grad();
		net->layers(1)->grad();
		net->layers(0)->grad();

		cacu_copy((convs[0])->in_data<blob>()->s_diff(), temp_d1->count(), ((blob*)element_sum1->get_oblob())->s_diff());
		//cacu_copy(temp_d1->s_data(), temp_d1->count(), (convs[0])->in_data<blob>()->s_data());
		element_sum1->grad();

		cacu_scalex(deconv3->output_blob()->s_diff(), deconv3->output_blob()->count(), deconv_add3);
		cacu_copy(temp_g3->s_data(), temp_g3->count(), deconv3->input_blobs()->asblob(0)->s_data());
		deconv3->backward_propagate();
		cacu_scalex(deconv2->output_blob()->s_diff(), deconv2->output_blob()->count(), deconv_add2);
		cacu_copy(temp_g2->s_data(), temp_g2->count(), deconv2->input_blobs()->asblob(0)->s_data());
		deconv2->backward_propagate();
		cacu_scalex(deconv1->output_blob()->s_diff(), deconv1->output_blob()->count(), deconv_add1);
		cacu_copy(temp_g1->s_data(), temp_g1->count(), deconv1->input_blobs()->asblob(0)->s_data());
		deconv1->backward_propagate();


		//sgd->updates(i);
		net->_RESET_WEIGHT_DIFF();

		sgd1->updates(i);
		sgd2->updates(i);
		sgd3->updates(i);

		timer->end();

		if(i % 10 == 0){

			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, sgd->lr(), timer->get_time_span() / 1000);
			LOG_INFO("predict loss: %f", predict_loss);
			((softmax_with_loss_op*)net->get_op(net->op_count() - 1))->echo();
			//net->get_op(10)->echo();
			//net->get_op(10)->echo();

			logger << ((softmax_with_loss_op*)net->get_op(net->op_count() - 1))->loss() << endl;
			logger.flush();
			/*
			cout << "rawdata:";
			cacu_print(net->get_op(0)->in_data<blob>()->s_data(), 10);
			cout << "visualized:";
			cacu_print(net->get_op(8)->out_data<blob>()->s_data(), 10);
			cout << "semantic:";
			cacu_print(net->get_op(9)->out_data<blob>()->s_data(), 10);
			//if (net->get_op(8)->out_data<blob>()->s_data()[0] + net->get_op(9)->out_data<blob>()->s_data()[0] - net->get_op(0)->in_data<blob>()->s_data()[0] > 0.00001)
			//	LOG_DEBUG("something is wrong!");
			//*/
		}

		if(i % 4000 == 0){
			sgd->set_lr_iter(0.1f);
			sgd1->set_lr_iter(0.1f);
			sgd2->set_lr_iter(0.1f);
			sgd3->set_lr_iter(0.1f);
		}
		/*
		if (i % 1000 == 0)
		{
			ostringstream oss;
			oss << "/home/haofang/experiment/cifar10/model_" << i << ".model";
			net->save_weights(oss.str());
		}
		*/

	}
	LOG_INFO("optimization is done!");
	net->save_weights(rootpath + "cifar10_quick_test.model");
	deconv1->save_weights(rootpath + "deconv1.model");
	deconv2->save_weights(rootpath + "deconv2.model");
	deconv3->save_weights(rootpath + "deconv3.model");

	vector<vec_t>().swap(full_data);
	vector<vec_i>().swap(full_label);
	logger.close();
	delete net;
	delete sgd;

	delete timer;

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif

