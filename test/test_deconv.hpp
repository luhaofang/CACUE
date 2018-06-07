/*
 * test_deconv.hpp
 *
 *  Created on: May 14, 2018
 *      Author: haofang
 */

#ifndef TEST_DECONV_HPP_
#define TEST_DECONV_HPP_


#include "../cacu/cacu.h"

#include "../cacu/user_ops/deconvolution_op.hpp"

#include <time.h>

#include "../tools/serializer_utils.h"

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

TEST_CASE("deconv") {
	SECTION("deconv functions test"){
	//cuda_set_device(2);
	set_rand_seed();
	blob_base *b = new blob(1,16,32,32,1,train);
	((blob *)b)->load_from("/home/haofang/git/mtcnn-caffe/demo/testdata.txt");
	data_args *args = new data_args(3,3,1,1,16);
	//cacu_bprint(((blob*)b));

	deconvolution_op *op_ = new deconvolution_op(b,args);
	op_->set_weight_init_type(constant,1);
	cacu_print(op_->get_weight(0)->s_data(),op_->get_weight(0)->count());
	op_->infer();
	op_->out_data<blob>()->set_diff(1);
	op_->grad();
	op_->out_data<blob>()->blob_size();
	LOG_DEBUG("diff");
	//cacu_bprint(op_->in_data<blob>());
	LOG_DEBUG("weight");
	//cacu_bprint(op_->get_weight(0));

	//blob *data_ = new blob(1,16,32,32,0,train);
	//data_->set_init_type(gaussian,1);
	serializer::blob_serialize(op_->in_data<blob>(),"/home/haofang/git/mtcnn-caffe/demo/cacudata.txt", train);
}

}



#endif /* TEST_DECONV_HPP_ */
