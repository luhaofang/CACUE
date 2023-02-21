/*
 * test_deconv.hpp
 *
 *  Created on: May 14, 2018
 *      Author: haofang
 */

#ifndef TEST_DECONV_HPP_
#define TEST_DECONV_HPP_


#include "../cacu/cacu.h"

#include "../cacu/user_ops/deconvolution_op.h"

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

	deconvolution_op *op_ = (deconvolution_op*)operator_factory::create_op(CACU_DECONVOLUTION, bs, args, NULL);
	op_->get_weight(0)->load_from("/Users/seallhf/Desktop/w.txt");

	//op_->set_weight_init_type(constant,1);
	//cacu_print(op_->get_weight(0)->s_data(),op_->get_weight(0)->count());
	//for(int i = 0 ; i < 10; ++i)
	op_->infer();
	//op_->out_data<blob>()->set_diff(1);
	blob *bf = new blob(1,32,256,256,0,train);
	bf->load_from("/Users/seallhf/Desktop/o_blob.txt");
	cacu_copy(bf->s_data(),bf->count(),op_->out_data<blob>()->s_diff());
	op_->derivative();
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

TEST_CASE("conv_trans") {
	SECTION("deconv functions test"){
	//cuda_set_device(2);
	set_rand_seed();
	blob_base *b = new blob(1,3,5,5,1,train);
	//((blob *)b)->load_from("/Users/seallhf/Desktop/s_blob.txt");
	//LOG_DEBUG("fuck");
	data_args *args = new data_args(9,4,1,0,3);
	//cacu_bprint(((blob*)b));

	blobs *bs = new blobs();
	bs->push_back(b);

	conv_transpose_op *op_ = (conv_transpose_op*)operator_factory::create_op(CACU_CONV_TRANS, bs, args, NULL);
	op_->get_weight(0)->set_init_type(constant, 1);

	//op_->set_weight_init_type(constant,1);
	//cacu_print(op_->get_weight(0)->s_data(),op_->get_weight(0)->count());
	//for(int i = 0 ; i < 10; ++i)
	op_->infer();
	//op_->out_data<blob>()->set_diff(1);
	blob *outdata = op_->out_data<blob>();
	outdata->blob_size();

	cacu_print(outdata->s_data(),outdata->count());

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

	xdiscriminator->top_op<sigmoid_with_loss_op>()->in_datas()->asbinblob(1)->set_data(1);

	xnet->forward_propagate();

	xdiscriminator->top_op<sigmoid_with_loss_op>()->echo();

	LOG_DEBUG("backward");

	xnet->backward_propagate();


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
	xdsgd->set_beta1(0.5);
	xdsgd->set_weight_decay(0.01);

	for(int i = 0 ; i < 10; ++i){
		xnet->forward_propagate();

		LOG_DEBUG("backward");

		xsuspicious->set_diff(0.1);

		xnet->backward_propagate();

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
	dsgd->set_beta1(0.5);
	dsgd->set_epsilon(0.00001);
	dsgd->set_weight_decay(0.01);

	adam_solver *gsgd = new adam_solver(gnet);
	gsgd->set_lr(0.0002);
	gsgd->set_beta1(0.5);
	gsgd->set_weight_decay(0.01);
	gsgd->set_epsilon(0.00001);

	for(int i = 1 ; i <= 10; ++i){

		readimg(xghead->s_data(),("/Users/seallhf/Documents/datasets/cyclegan/datasets/horse2zebra/trainA/n02381460_2.jpg"));
		readimg(xdhead->s_data(),("/Users/seallhf/Documents/datasets/cyclegan/datasets/horse2zebra/trainB/n02391049_15.jpg"));

		xdiscriminator->top_op<sigmoid_with_loss_op>()->in_datas()->asbinblob(1)->set_data(1);
		dsgd->train_iter(i);
		xdiscriminator->top_op<sigmoid_with_loss_op>()->echo();

		gnet->forward_propagate();

		//LOG_DEBUG("fake image!");

		//xdhead->copy_blob(xsuspicious);
		cacu_copy(xsuspicious->s_data(),xsuspicious->count(),xdhead->s_data());
		cacu_print(xsuspicious->s_data(), 100);
		xdiscriminator->top_op<sigmoid_with_loss_op>()->in_datas()->asbinblob(1)->set_data(0);
		dsgd->train_iter(i);
		xdiscriminator->top_op<sigmoid_with_loss_op>()->echo();

		//cacu_print(xdhead->s_diff(), 100);

		cacu_copy(xdhead->s_diff(),xsuspicious->count(),xsuspicious->s_diff());

		//LOG_DEBUG("backward");

		gsgd->update_direction(maximize);

		gnet->backward_propagate();

		gsgd->updates(i);
	}

}
}

TEST_CASE("conv_trans_op") {
	SECTION("deconv functions test"){
	//cuda_set_device(2);
	set_rand_seed();

	blob *datab = new blob(1, 1, 6, 6, 1,train);
	blobs *datasb = new blobs();
	datasb->push_back(datab);
	cacu_print(datab->s_data(),datab->count());
	cacu_op *deconv = new cacu_op(CACU_DECONVOLUTION, new data_args(1, 4, 2, 1, 1), train);
	deconv->get_param(0)->set_init_type(constant,1);
	deconv->forward(datasb);
	cacu_print(deconv->out_datas()->asblob(0)->s_data(),deconv->out_datas()->at(0)->count());
	deconv->out_datas()->asblob(0)->blob_size();
	deconv->out_datas()->asblob(0)->set_diff(1);
	deconv->backward();
	cacu_print(datab->s_diff(), datab->count());
	datab->_RESET_DIFF();

	cacu_op *conv_trans = new cacu_op(CACU_CONV_TRANS, new data_args(1, 4, 2, 1, 1), train);
	conv_trans->get_param(0)->set_init_type(constant,1);
	conv_trans->forward(datasb);
	cacu_print(conv_trans->out_datas()->asblob(0)->s_data(),conv_trans->out_datas()->at(0)->count());
	conv_trans->out_datas()->asblob(0)->blob_size();
	conv_trans->out_datas()->asblob(0)->set_diff(1);
	conv_trans->backward();
	cacu_print(datab->s_diff(), datab->count());

}
}

TEST_CASE("test_deconv") {
	SECTION("deconv functions test"){
	//cuda_set_device(2);
	//set_rand_seed();

	blob *datab = new blob(8, 1, 1, 1, 1, train);
	datab->set_init_type(uniform, 1);

	bin_blob *databin = new bin_blob(8, 1, 1, 1, 1, test);
	databin->set_data(1);

	//serializer::blob_serialize(datab,"/Users/seallhf/Desktop/data.txt",train);

	//weight *w = new weight("test", 3, 32, 32, 32, train);

	blobs *bs = new blobs();
	bs->push_back(datab);
	bs->push_back(databin);

	data_args *args = new data_args(3, 3, 1, 1, 3);

	datab->load_from("/Users/seallhf/Desktop/data.txt");

	//cacu_print(datab->s_data(),datab->count());

	sigmoid_with_loss_op *op_ = (sigmoid_with_loss_op*)operator_factory::create_op(CACU_SIGMOID_LOSS, bs, NULL, NULL);
	//op_->set_weight_init_type(gaussian,0.01);
	//op_->set_bias_init_type(constant,0.1);
	//serializer::blob_serialize(op_->get_weight(0),"/Users/seallhf/Desktop/w.txt",train);
	//serializer::blob_serialize(op_->get_weight(1),"/Users/seallhf/Desktop/b.txt",train);
	//op_->set_is_use_global_stats(false);
	//op_->get_weight(0)->load_from("/Users/seallhf/Desktop/w.txt");
	//op_->get_weight(1)->load_from("/Users/seallhf/Desktop/b.txt");
	op_->infer();

	op_->out_data<blob>()->set_diff(1.0);
	//cacu_print(op_->out_data<blob>()->s_data(), op_->out_data<blob>()->count());

	op_->derivative();
	//cacu_print(op_->in_data<blob>()->s_diff(), op_->in_data<blob>()->count());

}
}


TEST_CASE("create_conv_data") {
	SECTION("deconv functions test"){
	//cuda_set_device(2);
	set_rand_seed();

	blob *datab = new blob(32, 3, 3, 3, 1, train);
	datab->set_init_type(uniform, 1);
	cacu_print(datab->s_data(),datab->count());

	blob *data = new blob(1, 3, 32, 32, 1, train);
	data->set_init_type(uniform, 1);

	blob *bias = new blob(3, 1, 1, 1, 1, train);
	bias->set_init_type(uniform, 1);

	serializer::blob_serialize(datab,"/Users/seallhf/Desktop/w.txt",train);
	serializer::blob_serialize(data,"/Users/seallhf/Desktop/d.txt",train);
	serializer::blob_serialize(bias,"/Users/seallhf/Desktop/b.txt",train);

}
}

TEST_CASE("test_sn") {
        SECTION("deconv functions test"){
                blob *datab = new blob(1, 3, 32, 32, 1, train);
                //datab->set_init_type(uniform, 1);
                //serializer::blob_serialize(datab,"/Users/seallhf/Desktop/data.txt",train);
                //weight *w = new weight("test", 3, 32, 32, 32, train);
                blobs *bs = new blobs();
                bs->push_back(datab);                           //
                data_args *args = new data_args(32, 3, 1, 1, 3);
                datab->load_from("/Users/seallhf/Desktop/d.txt");
                //cacu_print(datab->s_data(),datab->count());
                sn_convolution_op *op_ = new sn_convolution_op(bs, args);
                //op_->set_is_use_global_stats(false);
                //deconvolution_op *op_ = new deconvolution_op(bs, args);
                //op_->set_weight_init_type(gaussian,0.01);
                //op_->set_bias_init_type(constant,0.1);
                op_->get_weight(0)->load_from("/Users/seallhf/Desktop/w.txt");
                op_->set_is_use_bias(false);
                cacu_print("w",op_->get_weight(0)->s_data(), op_->get_weight(0)->count());
                //op_->get_weight(1)->load_from("/home/luhaofang/b.txt");
                op_->infer();
                op_->out_data<blob>()->set_diff(1);
                cacu_print("wb",op_->get_weight(0)->s_data(), op_->get_weight(0)->count());
                //cacu_print(op_->out_data<blob>()->s_data(), op_->out_data<blob>()->count());
                op_->derivative();
                cacu_print(op_->in_data<blob>()->s_diff(), op_->in_data<blob>()->count());
        }
}


TEST_CASE("test_1x1col") {
        SECTION("deconv functions test"){
                blob *datab = new blob(1, 2, 3, 3, 1, test);
                blob *datao = new blob(1, 2, 2*1, 2*1, 1, test);
                float_t d[] = {1,2,1,2,1,2,0,1,1,0,1,1,1,2,1,2,1,2};
                cacu_copy(d, datab->count(), datab->s_data());
                cacu_img2col_pad(datab->s_data(),1,1,2,3,3,2,2,2,0,0,datao->s_data());
                cacu_print(datao->s_data(),datao->count());
                cacu_col2img_pad(datao->s_data(),1,1,2,3,3,2,2,2,0,0,datab->s_data());
				cacu_print(datab->s_data(),datab->count());


        }
}


#endif /* TEST_DECONV_HPP_ */
