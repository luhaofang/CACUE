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
#include "../example/cycle_gan/models.h"
#include "../example/frcnn/tools_frcnn.h"
#include "../cacu/solvers/adam_solver.h"


#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

TEST_CASE("deconv") {
	SECTION("deconv functions test"){
	//cuda_set_device(2);
	set_rand_seed();
	LOG_DEBUG("fuck");
	blob_base *b = new blob(1,64,128,128,0,train);
	((blob *)b)->load_from("/Users/seallhf/Desktop/s_blob.txt");
	LOG_DEBUG("fuck");
	data_args *args = new data_args(32,4,2,1,64);
	//cacu_bprint(((blob*)b));

	blobs *bs = new blobs();
	bs->push_back(b);

	deconvolution_op *op_ = new deconvolution_op(bs,args);
	op_->get_weight(0)->load_from("/Users/seallhf/Desktop/w.txt");

	//op_->set_weight_init_type(constant,1);
	//cacu_print(op_->get_weight(0)->s_data(),op_->get_weight(0)->count());
	//for(int i = 0 ; i < 10; ++i)
	op_->infer();
	//op_->out_data<blob>()->set_diff(1);
	blob *bf = new blob(1,32,256,256,0,train);
	bf->load_from("/Users/seallhf/Desktop/o_blob.txt");
	cacu_copy(bf->s_data(),bf->count(),op_->out_data<blob>()->s_diff());
	op_->grad();
	op_->out_data<blob>()->blob_size();
	LOG_DEBUG("diff");
	//cacu_bprint(op_->in_data<blob>());
	LOG_DEBUG("weight");
	//cacu_bprint(op_->get_weight(0));
	cacu_print(op_->out_data<blob>()->s_data(),100);
	//blob *data_ = new blob(1,16,32,32,0,train);
	//data_->set_init_type(gaussian,1);
	//serializer::blob_serialize(op_->in_data<blob>(),"", train);
}

}

TEST_CASE("discriminator") {
	SECTION("deconv functions test"){
	//cuda_set_device(2);
	set_rand_seed();

	network *xnet = new network();
	layer_block *xdiscriminator = create_discriminator_cycleGan(1, 64, train);
	*xnet << xdiscriminator;

	blob* xhead = xdiscriminator->layers(0)->get_head_op()->in_data<blob>();

	readimg(xhead->s_data(),("/Users/seallhf/Documents/datasets/cyclegan/datasets/horse2zebra/trainA/n02381460_2.jpg"));

	xdiscriminator->top_op<sigmoid_with_loss_op>()->in_datas()->astype<bin_blob>(1)->set_data(1);

	xnet->forward_propagate();

	xdiscriminator->top_op<sigmoid_with_loss_op>()->echo();

	LOG_DEBUG("backward");

	xnet->back_propagate();


}
}


TEST_CASE("generator") {
	SECTION("deconv functions test"){
	//cuda_set_device(2);
	set_rand_seed();


	network *xnet = new network();
	layer_block *xgenerator = create_generator_cycleGan(1, 32, train);
	*xnet << xgenerator;

	blob* xhead = xgenerator->layers(0)->get_head_op()->in_data<blob>();

	blob *xsuspicious = (blob*)xgenerator->get_oblob();

	readimg(xhead->s_data(),("/Users/seallhf/Documents/datasets/cyclegan/datasets/horse2zebra/trainA/n02381460_2.jpg"));

	adam_solver *xdsgd = new adam_solver(xnet);
	xdsgd->set_lr(0.0002);
	xdsgd->set_alpha(0.5);
	xdsgd->set_weight_decay(0.01);

	for(int i = 0 ; i < 10; ++i){
		xnet->forward_propagate();

		LOG_DEBUG("backward");

		xsuspicious->set_diff(0.1);

		xnet->back_propagate();

		xdsgd->updates(i);
	}

}
}

TEST_CASE("training_test") {
	SECTION("deconv functions test"){
	//cuda_set_device(2);
	set_rand_seed();

	network *gnet = new network();
	network *dnet = new network();
	layer_block *xdiscriminator = create_discriminator_cycleGan(1, 64, train);
	*dnet << xdiscriminator;
	blob* xdhead = xdiscriminator->layers(0)->get_head_op()->in_data<blob>();

	layer_block *xgenerator = create_generator_cycleGan(1, 32, train);
	*gnet << xgenerator;
	blob* xghead = xgenerator->layers(0)->get_head_op()->in_data<blob>();

	blob *xsuspicious = (blob*)xgenerator->get_oblob();

	adam_solver *dsgd = new adam_solver(dnet);
	dsgd->set_lr(0.0002);
	dsgd->set_alpha(0.5);
	dsgd->set_epsilon(0.00001);
	dsgd->set_weight_decay(0.01);

	adam_solver *gsgd = new adam_solver(gnet);
	gsgd->set_lr(0.0002);
	gsgd->set_alpha(0.5);
	gsgd->set_weight_decay(0.01);
	gsgd->set_epsilon(0.00001);

	for(int i = 1 ; i <= 10; ++i){

		readimg(xghead->s_data(),("/Users/seallhf/Documents/datasets/cyclegan/datasets/horse2zebra/trainA/n02381460_2.jpg"));
		readimg(xdhead->s_data(),("/Users/seallhf/Documents/datasets/cyclegan/datasets/horse2zebra/trainB/n02391049_15.jpg"));

		xdiscriminator->top_op<sigmoid_with_loss_op>()->in_datas()->astype<bin_blob>(1)->set_data(1);
		dsgd->train_iter(i);
		xdiscriminator->top_op<sigmoid_with_loss_op>()->echo();

		gnet->forward_propagate();

		//LOG_DEBUG("fake image!");

		//xdhead->copy_blob(xsuspicious);
		cacu_copy(xsuspicious->s_data(),xsuspicious->count(),xdhead->s_data());
		cacu_print(xsuspicious->s_data(), 100);
		xdiscriminator->top_op<sigmoid_with_loss_op>()->in_datas()->astype<bin_blob>(1)->set_data(0);
		dsgd->train_iter(i);
		xdiscriminator->top_op<sigmoid_with_loss_op>()->echo();

		//cacu_print(xdhead->s_diff(), 100);

		cacu_copy(xdhead->s_diff(),xsuspicious->count(),xsuspicious->s_diff());

		//LOG_DEBUG("backward");

		gsgd->update_direction(maximize);

		gnet->back_propagate();

		gsgd->updates(i);
	}

}
}



#endif /* TEST_DECONV_HPP_ */
