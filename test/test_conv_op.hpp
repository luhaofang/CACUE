#ifndef TEST_CONV_OP_HPP_
#define TEST_CONV_OP_HPP_


#include "../cacu/cacu.h"

#include "../cacu/framework/blob.h"
#include "../cacu/tensor/tensor.h"
#include "../cacu/utils/data_printer.h"
#include "../cacu/tensor/utils/device_data_utils.h"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

TEST_CASE("conv")
{
	SECTION("conv functions test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
		cuda_set_device(0);
//		set_rand_seed();

//		create_cublas_handle();
		time_utils *timer = new time_utils();
		blob *datab = new blob(1, 4, 16, 16, 1, train);
		blobs *datasb = new blobs();
		datasb->push_back(datab);
//		cacu_print(datab->s_data(),datab->count());
		data_args *args = new data_args(8, 3, 1, 1, 4);
		equalized_convolution_op *conv = new equalized_convolution_op(datasb, (data_args *&)args);
		conv->set_use_wscale(true);
//		conv->set_group(4);
		conv->std_init_w();
//		conv->set_is_use_bias(false);
//		conv->get_weight(0)->set_init_type(gaussian,0.1);
//		conv->get_weight(1)->set_init_type(gaussian,0.1);
		cacu_print(conv->get_weight(0)->s_data(), conv->get_weight(0)->count());
		timer->start();
		for(int i = 0 ; i < 1; ++i)
			conv->infer();
		timer->end();
		cacu_print(conv->get_weight(0)->s_data(), conv->get_weight(0)->count());
		cacu_print(conv->get_weight(0)->s_diff(), conv->get_weight(0)->count());
		LOG_DEBUG("%ld ms", timer->get_time_span() / 1000);

//		datab->set_tensor_desc(100, datab->channel(), datab->width(), datab->height());
//		conv->infer();
//		LOG_DEBUG("%ld ms", timer->get_time_span() / 1000);

		cacu_print(conv->out_datas()->asblob(0)->s_data(), conv->out_datas()->at(0)->count());
#endif
#endif
	}

}


TEST_CASE("ip_equalized")
{
	SECTION("conv functions test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
		cuda_set_device(0);
//		set_rand_seed();

//		create_cublas_handle();
		time_utils *timer = new time_utils();

		cuda_set_device(0);
		set_rand_seed();
		blob_base *b = new blob(1,3,1,1,1,train);
		//((blob *)b)->load_from("/home/haofang/git/mtcnn-caffe/demo/testdata.txt");
		data_args *args = new data_args(1,1,1,0,3);
		network *net = new network();

		layer *conv1 = new layer(args);
		conv1->op<equalized_inner_product_op>(b);
		conv1->get_op<equalized_inner_product_op>(0)->set_is_use_bias(true);
		conv1->get_op<equalized_inner_product_op>(0)->set_use_wscale(true);
		conv1->get_op<equalized_inner_product_op>(0)->std_init_w();
		*net << conv1;

		equalized_inner_product_op * conv = conv1->get_op<equalized_inner_product_op>(0);


		blobs *datas = new blobs();
		datas->push_back(conv->out_data<blob>());
		datas->push_back(new blob(1,1,1,1,1,train));

		adam_solver *adam = new adam_solver(net);
		adam->set_regularize(cacu::regularize_type::L2);
	//	gsgd->load_param(dir_path + "g_solver.txt");
		adam->set_beta1(0.5);
		adam->set_beta2(0.999);
		adam->set_lr(0.1);
		adam->set_weight_decay(0.000);

		mse_loss_op *loss_ = new mse_loss_op(datas);

		cacu_print(conv->get_weight(0)->s_data(), conv->get_weight(0)->count());
		timer->start();
		for(int i = 0 ; i < 10; ++i) {
			net->forward_propagate();

			std::cout << i << ":  " << "+================================" << std::endl;
			loss_->infer();

			loss_->derivative();

			net->backward_propagate();

			cacu_bprint(conv->get_weight(0), train);

			adam->updates(i + 1);

			LOG_DEBUG("diff");

			cacu_bprint(conv->out_data<blob>(), test);
			loss_->echo();
			cacu_bprint(conv->get_weight(0), test);
		}
		timer->end();
		cacu_print(conv->get_weight(0)->s_data(), conv->get_weight(0)->count());
		cacu_print(conv->get_weight(0)->s_diff(), conv->get_weight(0)->count());
		LOG_DEBUG("%ld ms", timer->get_time_span() / 1000);

//		datab->set_tensor_desc(100, datab->channel(), datab->width(), datab->height());
//		conv->infer();
//		LOG_DEBUG("%ld ms", timer->get_time_span() / 1000);

		cacu_print(conv->out_datas()->asblob(0)->s_data(), conv->out_datas()->at(0)->count());
#endif
#endif
	}

}


TEST_CASE("fc_test")
{
	SECTION("conv functions test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
		cuda_set_device(0);
//		set_rand_seed();

//		create_cublas_handle();
		time_utils *timer = new time_utils();
		blob *datab = new blob(1, 4, 6, 6, 1, train);
		datab->set_init_type(gaussian,0.1);
		blobs *datasb = new blobs();
		datasb->push_back(datab);
//		cacu_print(datab->s_data(),datab->count());
		data_args *args = new data_args(8, 6, 1, 0, 4);
		convolution_op *conv = new convolution_op(datasb, (data_args *&)args);
//		conv->set_group(4);
//		conv->set_is_use_bias(false);
		conv->get_weight(0)->set_init_type(gaussian,1);
		conv->get_weight(1)->set_init_type(gaussian,1);
		timer->start();
		for(int i = 0 ; i < 100; ++i)
			conv->infer();
		timer->end();
		LOG_DEBUG("conv: %ld ms", timer->get_time_span() / 1000);

		inner_product_op *ip = new inner_product_op(datasb, (data_args *&)args);
//		ip->set_group(4);
//		conv->set_is_use_bias(false);
		cacu_copy(conv->get_weight(0)->s_data(), conv->get_weight(0)->count(), ip->get_weight(0)->s_data());
		cacu_copy(conv->get_weight(1)->s_data(), conv->get_weight(1)->count(), ip->get_weight(1)->s_data());
		timer->start();
		for(int i = 0 ; i < 100; ++i)
			ip->infer();
		timer->end();
		LOG_DEBUG("ip: %ld ms", timer->get_time_span() / 1000);

//		datab->set_tensor_desc(100, datab->channel(), datab->width(), datab->height());
//		conv->infer();
//		LOG_DEBUG("%ld ms", timer->get_time_span() / 1000);

		cacu_print(conv->out_datas()->asblob(0)->s_data(), conv->out_datas()->at(0)->count());

		cacu_print(conv->out_datas()->asblob(0)->s_data(), conv->out_datas()->at(0)->count());
#endif
#endif
	}

}


TEST_CASE("deconv2")
{
	SECTION("deconv functions test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
		cuda_set_device(0);
//		set_rand_seed();

//		create_cublas_handle();
		time_utils *timer = new time_utils();
		blob *datab = new blob(10, 4, 6, 6, 1, train);
		blobs *datasb = new blobs();
		datasb->push_back(datab);
//		cacu_print(datab->s_data(),datab->count());
		data_args *args = new data_args(8, 3, 2, 1, 4);
		conv_transpose_op *conv = new conv_transpose_op(datasb, (data_args *&)args);
//		conv->set_group(4);
//		conv->set_is_use_bias(false);
		conv->get_weight(0)->set_init_type(gaussian,0.1);
		conv->get_weight(1)->set_init_type(gaussian,0.1);
		timer->start();
		for(int i = 0 ; i < 1; ++i){
			conv->infer();
			conv->out_datas()->asblob(0)->set_diff(1);
			conv->derivative();
		}
		timer->end();
		LOG_DEBUG("%ld ms", timer->get_time_span() / 1000);

//		datab->set_tensor_desc(100, datab->channel(), datab->width(), datab->height());
//		conv->infer();
//		LOG_DEBUG("%ld ms", timer->get_time_span() / 1000);

		cacu_print("outdata", conv->out_datas()->asblob(0)->s_data(), conv->out_datas()->at(0)->count());
		cacu_print("indata", conv->in_datas()->asblob(0)->s_diff(), conv->in_datas()->at(0)->count());
		cacu_print("w", conv->get_weight(0)->s_diff(), conv->get_weight(0)->count());
		cacu_print("bias", conv->get_weight(1)->s_diff(), conv->get_weight(1)->count());
#endif
#endif
	}

}

TEST_CASE("bn")
{
	SECTION("bn functions test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
		cuda_set_device(0);
//		set_rand_seed();
#endif
#endif

//		create_cublas_handle();
		time_utils *timer = new time_utils();
		blob *datab = new blob(1, 3, 6, 6, 1, train);
		datab->set_init_type(gaussian, 1);
//		cacu_scalex(datab->p_data(1), datab->length(), 2);
		blobs *datasb = new blobs();
		datasb->push_back(datab);

		blob *datab1 = new blob(2, 3, 6, 6, 1, train);
		datab1->set_init_type(gaussian, 1);
		bin_blob *label = new bin_blob(2, 1, 1, 1, 1, train);
		vec_i v(2);
		v[0] = 0;
		v[1] = 1;
		label->copy2data(v);
//		cacu_scalex(datab->p_data(1), datab->length(), 2);
		blobs *datasb1 = new blobs();

		datasb1->push_back(datab1);
		datasb1->push_back(label);

		cacu_copy(datab1->p_data(1), datab1->length(), datab->s_data());
//		cacu_print(datab->s_data(),datab->count());
		data_args *args = new data_args(3, 3, 1, 0, 3);
		op_args *o_args = new op_args(2);
		conditional_in_op *in = new conditional_in_op(datasb1, (op_args *&)o_args, (data_args *&)args);
		in->set_phase(test);

		batch_normalize_op *bn = new batch_normalize_op(datasb, (data_args *&)args);
//		bn->get_storage_blob(4)->set_init_type(gaussian, 0.1);
//		bn->get_storage_blob(3)->set_init_type(gaussian, 0.1);
//		bn->get_storage_blob(2)->set_init_type(gaussian, 0.1);
//		bn->get_storage_blob(1)->set_init_type(gaussian, 0.1);
//		conv->set_is_use_bias(false);
		timer->start();
		for(int i = 0 ; i < 10; ++i){
			in->infer();

		}
		bn->infer();
		timer->end();
		bn->out_datas()->asblob(0)->set_diff(0.1);
		bn->derivative();
		in->out_datas()->asblob(0)->set_diff(0.1);
		in->derivative();
		LOG_DEBUG("%ld ms", timer->get_time_span() / 1000);

//		datab->set_tensor_desc(100, datab->channel(), datab->width(), datab->height());
//		conv->infer();
//		LOG_DEBUG("%ld ms", timer->get_time_span() / 1000);

//		cacu_print(bn->out_datas()->asblob(0)->s_data(), bn->out_datas()->at(0)->count());
//		cacu_print(datab->s_diff(), datab->count());
		cacu_print(in->out_datas()->asblob(0)->p_data(1), in->out_datas()->at(0)->length());
//		cacu_print(datab1->s_diff(), datab1->count());

	}

}



TEST_CASE("pooling")
{
	SECTION("conv functions test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
		cuda_set_device(0);
//		set_rand_seed();

//		create_cublas_handle();

		blob *datab = new blob(20, 3, 10, 10, 1, train);
		datab->set_init_type(gaussian, 1);
		blobs *datasb = new blobs();
		datasb->push_back(datab);
		cacu_print(datab->s_data(),datab->count());
		data_args *args = new data_args(3, 3, 2, 1, 3);
		max_pooling_op *max = new max_pooling_op(datasb, (data_args *&)args);
		for(int i = 0 ; i < 1; ++i)
			max->infer();
		cacu_print("max", max->out_datas()->asblob(0)->s_data(), max->out_datas()->at(0)->count());

		average_pooling_op *ave = new average_pooling_op(datasb, (data_args *&)args);
		for(int i = 0 ; i < 1; ++i)
			ave->infer();
		cacu_print("ave", ave->out_datas()->asblob(0)->s_data(), ave->out_datas()->at(0)->count());
#endif
#endif
	}

}



#endif
