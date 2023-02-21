/*
 * test_sigmoid_loss.hpp
 *
 *  Created on: May 30, 2018
 *      Author: haofang
 */

#ifndef TEST_SIGMOID_LOSS_HPP_
#define TEST_SIGMOID_LOSS_HPP_


#include "../cacu/cacu.h"

#include "../cacu/ops/sigmoid_with_loss_op.h"

#include <time.h>

#include "../tools/serializer_utils.h"

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

TEST_CASE("sloss") {
	SECTION("sloss functions test"){
#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif
#endif
	set_rand_seed();
	blob_base *b = new blob(1,3,3,3,1,train);
	//((blob *)b)->load_from("/home/haofang/git/mtcnn-caffe/demo/testdata.txt");
	data_args *args = new data_args(1,3,1,0,3);
	network *net = new network();

	layer *conv = new layer(args);
	conv->op<convolution_op>(b)->op<scalex_op>(new op_args(0.01));
	conv->get_op<convolution_op>(0)->set_is_use_bias(false);
	conv->get_op<convolution_op>(0)->set_weight_init_type(constant, 1);

	*net << conv;

	blobs *datas = new blobs();
	datas->push_back(conv->get_oblob());
	datas->push_back(new blob(1,1,1,1,1,train));

	adam_solver *adam = new adam_solver(net);
	adam->set_regularize(cacu::regularize_type::L2);
//	gsgd->load_param(dir_path + "g_solver.txt");
	adam->set_beta1(0.5);
	adam->set_beta2(0.999);
	adam->set_lr(0.1);
	adam->set_weight_decay(0.000);

	mse_loss_op *loss_ = new mse_loss_op(datas);

	for(int i = 0 ; i < 10; ++i){

		net->forward_propagate();

		std::cout << i << ":  " << "+================================" << std::endl;
		loss_->infer();

		loss_->derivative();

		net->backward_propagate();

		cacu_bprint(conv->get_op<convolution_op>(0)->get_weight(0), train);

		adam->updates(i + 1);

		LOG_DEBUG("diff");

		cacu_bprint(conv->oblob(), test);
		loss_->echo();
		cacu_bprint(conv->get_op<convolution_op>(0)->get_weight(0), test);
	}
	//cacu_bprint(op_->get_weight(0));

	//blob *data_ = new blob(1,16,32,32,0,train);
	//data_->set_init_type(gaussian,1);
	//serializer::blob_serialize(op_->out_data<blob>(),"/home/haofang/git/mtcnn-caffe/demo/cacudiff_l.txt", train);
}

}




#endif /* TEST_SIGMOID_LOSS_HPP_ */
