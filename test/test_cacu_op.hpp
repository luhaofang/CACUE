/*
 * test_cacu_op.hpp
 *
 *  Created on: May 14, 2018
 *      Author: haofang
 */

#ifndef TEST_CACU_OP_HPP_
#define TEST_CACU_OP_HPP_

#include "../cacu/cacu.h"

#include "../cacu/framework/cacu_op.h"

#include <time.h>

#include "../tools/serializer_utils.h"

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

TEST_CASE("cacu_op") {

	SECTION("cacu_op dynamic test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
	set_rand_seed();

	blob *datab = new blob(1, 4, 6, 6, 1,train);
	blobs *datasb = new blobs();
	datasb->push_back(datab);
	cacu_print(datab->s_data(),datab->count());
	cacu_op *conv = new cacu_op(CACU_CONVOLUTION, new data_args(3, 3, 1, 1, 4), train);
	conv->get_param(0)->set_init_type(gaussian,0.1);
	conv->forward(datasb);

	conv->out_datas()->asblob(0)->blob_size();
	cacu_print(conv->out_datas()->asblob(0)->s_data(),conv->out_datas()->at(0)->count());

	cacu_op *bn = new cacu_op(CACU_BATCH_NORMALIZE, new data_args(3, 2, 2, 0, 3), train);
	bn->forward(conv->out_datas());
	bn->out_datas()->asblob(0)->blob_size();

	cacu_print("up_pooling", bn->out_datas()->asblob(0)->s_data(),bn->out_datas()->at(0)->count());

	cacu_op *relu = new cacu_op(CACU_LEAKY_RELU, new op_args(0.2), train);
	relu->forward(bn->out_datas());
	cacu_print(relu->out_datas()->asblob(0)->s_data(),relu->out_datas()->at(0)->count());

	cacu_op *dropout = new cacu_op(CACU_DROPOUT, new op_args(0.5), train);
	dropout->forward(relu->out_datas());
	cacu_print(dropout->out_datas()->asblob(0)->s_data(),dropout->out_datas()->at(0)->count());

	dropout->backward();
	cacu_print("dropout_diff", dropout->in_datas()->asblob(0)->s_diff(),dropout->in_datas()->at(0)->count());
	relu->backward();
	cacu_print("relu_diff", relu->in_datas()->asblob(0)->s_data(),relu->in_datas()->at(0)->count());
	bn->backward();
	cacu_print("bn_diff", bn->in_datas()->asblob(0)->s_data(),bn->in_datas()->at(0)->count());
	conv->backward();
	cacu_print("conv_diff", conv->in_datas()->asblob(0)->s_data(),conv->in_datas()->at(0)->count());

	datab->resize(3, 3, 25, 25);
	datab->blob_size();
	datab->set_data(1);
	conv->forward(datasb);
	LOG_DEBUG("fuck");
	conv->out_datas()->asblob(0)->blob_size();
	cacu_print(conv->out_datas()->asblob(0)->s_data(),conv->out_datas()->at(0)->count());
	bn->forward(conv->out_datas());
	cacu_print(bn->out_datas()->asblob(0)->s_data(),bn->out_datas()->at(0)->count());
	relu->forward(bn->out_datas());
	cacu_print(relu->out_datas()->asblob(0)->s_data(),relu->out_datas()->at(0)->count());
	dropout->forward(relu->out_datas());
	cacu_print(dropout->out_datas()->asblob(0)->s_data(),dropout->out_datas()->at(0)->count());

#endif
#endif

	}
}

TEST_CASE("cacu_op_relu") {


	SECTION("cacu_op dynamic test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
	set_rand_seed();

	blob *datab = new blob(1, 4, 6, 6, 1,train);
	blobs *datasb = new blobs();
	datasb->push_back(datab);
	datab->set_init_type(gaussian,0.1);
	cacu_print("indata",datab->s_data(),datab->count());
	cacu_op *conv = new cacu_op(CACU_LEAKY_RELU, new op_args(0.1), train);
	conv->forward(datasb);
	cacu_print("outdata",conv->out_datas()->asblob(0)->s_data(),datab->count());
	conv->out_datas()->asblob(0)->set_diff(1);
	conv->backward();

	cacu_print("indiff",datab->s_diff(), datab->count());

#endif
#endif
	}

}

TEST_CASE("cacu_op_us") {
	SECTION("cacu_op dynamic test"){

	blob *datab = new blob(2, 3, 16, 16, 1,train);
	blobs *datasb = new blobs();
	datasb->push_back(datab);
	datab->set_init_type(gaussian, 1);
	cacu_print(datab->s_data(),datab->count());
	//cacu_op *conv = new cacu_op(CACU_CONVOLUTION, new data_args(3, 2, 2, 0, 3), train);
	data_args *_args = new data_args(3, 2, 2, 0, 3);
	conv_transpose_op *conv = new conv_transpose_op((blobs*&)datasb, (data_args *&)_args);
	conv->set_weight_init_type(constant, 1);
	conv->set_group(3);
	conv->set_is_use_bias(false);
	//serializer::blob_serialize(datab, "/Users/seallhf/Desktop/sb.txt", train);
	conv->infer();
	conv->out_datas()->asblob(0)->blob_size();
	cacu_print("forward",conv->out_datas()->asblob(0)->s_data(),conv->out_datas()->at(0)->count());

	conv->out_datas()->asblob(0)->set_diff(1);
	//cacu_print(conv->out_datas()->astype<blob>(0)->s_diff(),conv->out_datas()->at(0)->count());

	LOG_DEBUG("fuck");
	conv->derivative();
	cacu_print("backward",datab->s_diff(),datab->count());

}
}

TEST_CASE("cacu_op_ave") {

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif
#endif
	SECTION("cacu_op dynamic test"){

	blob *datab = new blob(2, 3, 16, 16, 1,train);
	blobs *datasb = new blobs();
	datasb->push_back(datab);
	datab->set_init_type(gaussian, 1);
	cacu_print(datab->s_data(),datab->count());
	//cacu_op *conv = new cacu_op(CACU_CONVOLUTION, new data_args(3, 2, 2, 0, 3), train);
	data_args *_args = new data_args(3, 3, 2, 0, 3);
	un_pooling_op *conv = new un_pooling_op((blobs*&)datasb, (data_args *&)_args);
	//conv->set_weight_init_type(constant, 1);
	//cacu_print(conv->get_weight(0)->s_data(), conv->get_weight(0)->count());
	//conv->set_is_use_bias(false);
	//serializer::blob_serialize(datab, "/Users/seallhf/Desktop/sb.txt", train);
	conv->infer();
	conv->out_datas()->asblob(0)->blob_size();
	cacu_print("forward",conv->out_datas()->asblob(0)->s_data(),conv->out_datas()->at(0)->count());

	conv->out_datas()->asblob(0)->set_diff(1);
	//cacu_print(conv->out_datas()->astype<blob>(0)->s_diff(),conv->out_datas()->at(0)->count());

	LOG_DEBUG("fuck");
	conv->derivative();
	cacu_print("backward",datab->s_diff(),datab->count());

}
}

TEST_CASE("cacu_norm") {
	SECTION("cacu_op dynamic test"){

	blob *datab = new blob(2, 3, 16, 16, 1, test);
	datab->set_init_type(uniform, 1);
	cacu_print(datab->s_data(), datab->count());

	float_t l1norm;
	cacu_norm(datab->s_data(), datab->count(), 0, norm_l2, &l1norm);
	LOG_DEBUG("%f", l1norm);
}
}

TEST_CASE("reg") {
	SECTION("cacu_op dynamic test"){

	blob *datab = new blob(2, 3, 16, 16, 1,train);
	blobs *datasb = new blobs();
	datasb->push_back(datab);
	datab->set_init_type(gaussian, 1);
	cacu_print(datab->s_data(),datab->count());
	//cacu_op *conv = new cacu_op(CACU_CONVOLUTION, new data_args(3, 2, 2, 0, 3), train);
	data_args *_args = new data_args(3, 2, 2, 0, 3);
	layer *l = new layer(_args);
	l->op<convolution_op>(datasb);
	//convolution_op *conv = cacu::factory<convolution_op>::produce("convolution_op", datasb, _args);
	LOG_DEBUG("%s", cacu::macro_factory::get_op_name(CACU_CONVOLUTION).c_str());
	l->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian, 1);
	convolution_op *conv = l->get_op<convolution_op>(0, CACU_CONVOLUTION);
	//cacu_print(conv->get_weight(0)->s_data(), conv->get_weight(0)->count());
	//conv->set_is_use_bias(false);
	//serializer::blob_serialize(datab, "/Users/seallhf/Desktop/sb.txt", train);
	//conv->op(datasb, conv->out_datas());
	//conv->out_datas()->asblob(0)->blob_size();
	l->operate();
	cacu_print("forward",conv->out_datas()->asblob(0)->s_data(),conv->out_datas()->at(0)->count());

	conv->out_datas()->asblob(0)->set_diff(1);
	//cacu_print(conv->out_datas()->astype<blob>(0)->s_diff(),conv->out_datas()->at(0)->count());

	LOG_DEBUG("fuck");
	conv->derivative();
	cacu_print("backward",datab->s_diff(),datab->count());

}
}

#endif /* TEST_DECONV_HPP_ */
