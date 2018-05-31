/*
 * test_sigmoid_loss.hpp
 *
 *  Created on: May 30, 2018
 *      Author: haofang
 */

#ifndef TEST_SIGMOID_LOSS_HPP_
#define TEST_SIGMOID_LOSS_HPP_


#include "../cacu/cacu.h"

#include "../cacu/ops/sigmoid_with_loss_op.hpp"

#include <time.h>

#include "../tools/serializer_utils.h"

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

TEST_CASE("sloss") {
	SECTION("sloss functions test"){
	//cuda_set_device(2);
	set_rand_seed();
	blob_base *b = new blob(1,1,1,1,1.6323,train);
	//((blob *)b)->load_from("/home/haofang/git/mtcnn-caffe/demo/testdata.txt");
	data_args *args = new data_args(1,3,1,1,16);
	//cacu_bprint(((blob*)b));

	//cacu_print(op_->get_weight(0)->s_data(),op_->get_weight(0)->count());

	blobs *datas = new blobs();
	datas->push_back(b);
	datas->push_back(new blob(1,1,1,1,1,train));
	sigmoid_with_loss_op *loss_ = new sigmoid_with_loss_op(datas,args);
	loss_->infer();
	loss_->echo();
	loss_->grad();

	LOG_DEBUG("diff");
	cacu_bprint(loss_->out_data<blob>(),test);
	cacu_bprint((blob*)loss_->in_datas()->at(0),train);
	LOG_DEBUG("weight");
	//cacu_bprint(op_->get_weight(0));

	//blob *data_ = new blob(1,16,32,32,0,train);
	//data_->set_init_type(gaussian,1);
	//serializer::blob_serialize(op_->out_data<blob>(),"/home/haofang/git/mtcnn-caffe/demo/cacudiff_l.txt", train);
}

}




#endif /* TEST_SIGMOID_LOSS_HPP_ */
