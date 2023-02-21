/*
 * test_functions.hpp
 *
 *  Created on: May 14, 2018
 *      Author: haofang
 */
/*

#ifndef TEST_FUNCTIONS_HPP_
#define TEST_FUNCTIONS_HPP_


#include "../cacu/cacu.h"

#include "../cacu/math/math.h"

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;


#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__

TEST_CASE("norm_functions") {
	SECTION("cacu_l1_normalization"){
		cuda_set_device(0);
		set_rand_seed();
		blob *data = new blob(3, 32, 16, 16, 0, test);
		data->set_init_type(gaussian, 0.1);
		cacu_l1_normalization_cuda(data->s_data(), data->count(), 1e-8);
		cacu_l1_normalization_cpu(data->s_data_cpu(), data->count(), 1e-8);
		vec_t temp(data->count());
		data->copy2data(temp);
		float_t acc_error = 0;
		for(int i = 0; i < data->count(); ++i)
			acc_error += abs(temp[i] - data->s_data_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);
		delete data;
	}

	SECTION("cacu_l1_normalization_grad"){
		cuda_set_device(0);
		set_rand_seed();
		blob *data = new blob(3, 32, 16, 16, 0, train);
		blob *grad = new blob(3, 32, 16, 16, 0, test);
		grad->set_init_type(gaussian, 0.1);
		data->set_init_type(gaussian, 0.1);
		cacu_l1_normalization_grad_cuda(data->s_data(), data->count(), 1e-8, grad->s_data(), data->s_diff());
		cacu_l1_normalization_grad_cpu(data->s_data_cpu(), data->count(), 1e-8, grad->s_data_cpu(), data->s_diff_cpu());
		vec_t temp(data->count());
		data->copy2diff(temp);
		float_t acc_error = 0;
		for(int i = 0; i < data->count(); ++i)
			acc_error += abs(temp[i] - data->s_diff_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);
	}

	SECTION("cacu_l2_normalization"){
		cuda_set_device(0);
		set_rand_seed();
		blob *data = new blob(3, 32, 16, 16, 0, test);
		data->set_init_type(gaussian, 0.1);
		cacu_l2_normalization_cuda(data->s_data(), data->count(), 1e-8);
		cacu_l2_normalization_cpu(data->s_data_cpu(), data->count(), 1e-8);
		vec_t temp(data->count());
		data->copy2data(temp);
		float_t acc_error = 0;
		for(int i = 0; i < data->count(); ++i)
			acc_error += abs(temp[i] - data->s_data_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);
		delete data;
	}

	SECTION("cacu_l2_normalization_grad"){
		cuda_set_device(0);
		set_rand_seed();
		blob *data = new blob(3, 32, 16, 16, 0, train);
		blob *grad = new blob(3, 32, 16, 16, 0, test);
		grad->set_init_type(gaussian, 0.1);
		data->set_init_type(gaussian, 0.1);
		cacu_l2_normalization_grad_cuda(data->s_data(), data->count(), 1e-8, grad->s_data(), data->s_diff());
		cacu_l2_normalization_grad_cpu(data->s_data_cpu(), data->count(), 1e-8, grad->s_data_cpu(), data->s_diff_cpu());
		vec_t temp(data->count());
		data->copy2diff(temp);
		float_t acc_error = 0;
		for(int i = 0; i < data->count(); ++i)
			acc_error += abs(temp[i] - data->s_diff_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);
	}
}
TEST_CASE("pooling_functions") {
	SECTION("max_pooling"){
		cuda_set_device(0);
		set_rand_seed();
		blob *data = new blob(1, 32, 16, 16, 0, train);
		blob *result = new blob(1, 32, 8, 8, 0, train);
		bin_blob *index_ = new bin_blob(1, 32, 8, 8, 0, test);
		data->set_init_type(gaussian, 0.1);
		cacu_max_pooling_cuda(data->s_data(), 3, 2, data->width(), data->height(), result->width(),
				result->height(), data->channel(), result->s_data(), index_->s_data());
		cacu_max_pooling_cpu(data->s_data_cpu(), 3, 2, data->width(), data->height(), result->width(),
				result->height(), data->channel(), result->s_data_cpu(), index_->s_data_cpu());
		vec_t temp(result->count());
		result->copy2data(temp);
		float_t acc_error = 0;
		for(int i = 0; i < result->count(); ++i)
			acc_error += abs(temp[i] - result->s_data_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);

		result->set_diff(0.1);
		cacu_max_pooling_grad_cuda(result->s_diff(), 3, 2, data->width(), data->height(), result->width(),
				result->height(), data->channel(), data->s_diff(), index_->s_data());
		cacu_max_pooling_grad_cpu(result->s_diff_cpu(), 3, 2, data->width(), data->height(), result->width(),
						result->height(), data->channel(), data->s_diff_cpu(), index_->_cpu());
		vec_t temp_diff(data->count());
		data->copy2diff(temp_diff);
		float_t acc_error = 0;
		for(int i = 0; i < data->count(); ++i)
			acc_error += abs(temp_diff[i] - data->s_diff_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);

		delete data, result, index_;
	}

	SECTION("ave_pooling"){
		cuda_set_device(0);
		set_rand_seed();
		blob *data = new blob(1, 32, 16, 16, 0, train);
		blob *result = new blob(1, 32, 8, 8, 0, train);
		data->set_init_type(gaussian, 0.1);
		cacu_average_pooling_cuda(data->s_data(), 3, 2, data->width(), data->height(), result->width(),
				result->height(), data->channel(), result->s_data());
		cacu_average_pooling_cpu(data->s_data_cpu(), 3, 2, data->width(), data->height(), result->width(),
						result->height(), data->channel(), result->s_data_cpu());
		vec_t temp(result->count());
		result->copy2data(temp);
		float_t acc_error = 0;
		for(int i = 0; i < result->count(); ++i)
			acc_error += abs(temp[i] - result->s_data_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);

		result->set_diff(0.1);
		cacu_average_pooling_grad_cuda(result->s_diff(), 3, 2, data->width(), data->height(), result->width(),
				result->height(), data->channel(), data->s_diff());
		cacu_average_pooling_grad_cpu(result->s_diff_cpu(), 3, 2, data->width(), data->height(), result->width(),
						result->height(), data->channel(), data->s_diff_cpu());
		vec_t temp_diff(data->count());
		data->copy2diff(temp_diff);
		float_t acc_error = 0;
		for(int i = 0; i < data->count(); ++i)
			acc_error += abs(temp_diff[i] - data->s_diff_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);

		delete data, result;
	}

	SECTION("img_col"){
		cuda_set_device(0);
		set_rand_seed();
		blob *data = new blob(1, 32, 16, 16, 0, test);
		blob *result = new blob(1, 32, 3*8, 3*8, 0, test);
		data->set_init_type(gaussian, 0.1);
		cacu_img2col_pad_cuda(data->s_data(), 3, 3, 2, data->width(), data->height(), data->channel(),
							8, 8, 1, 1, result->s_data());
		cacu_img2col_pad_cpu(data->s_data_cpu(), 3, 3, 2, data->width(), data->height(), data->channel(),
							8, 8, 1, 1, result->s_data_cpu());
		vec_t temp(result->count());
		result->copy2data(temp);
		float_t acc_error = 0;
		for(int i = 0; i < result->count(); ++i)
			acc_error += abs(temp[i] - result->s_data_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);

		cacu_col2img_pad_cuda(result->s_data(), 3, 3, 2, data->width(),data->height(),data->channel(),
				result->width(), result->height(), 1, 1, data->s_data());
		cacu_col2img_pad_cpu(result->s_data_cpu(), 3, 3, 2, data->width(),data->height(),data->channel(),
				result->width(), result->height(), 1, 1, data->s_data_cpu());
		vec_t temp_data(data->count());
		data->copy2data(temp_diff);
		float_t acc_error = 0;
		for(int i = 0; i < data->count(); ++i)
			acc_error += abs(temp_data[i] - data->s_data_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);

		delete data, result;
	}
}

TEST_CASE("batch_functions") {
	SECTION("cacu_sumbysize_width"){
		cuda_set_device(0);
		set_rand_seed();
		blob *data = new blob(3, 32, 16, 16, 0, test);
		blob * result = new blob(1,1,1,1,0,test);
		float_t m = 0;
		data->set_init_type(gaussian, 0.1);
		cacu_sumbysize_cuda(BYWIDTH, data->s_data(), data->count(),
				1, result->s_data(), 0, data->count());
		cacu_sumbysize_cpu(BYWIDTH, data->s_data_cpu(), data->count(),
				1, result->s_data_cpu(), 0, data->count());
		vec_t temp(result->count());
		result->copy2data(temp);
		float_t acc_error = 0;
		for(int i = 0; i < result->count(); ++i)
			acc_error += abs(temp[i] - result->s_data_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);
		delete data, result;
	}


	SECTION("cacu_sumbysize_height"){
		cuda_set_device(0);
		set_rand_seed();
		blob *data = new blob(3, 32, 16, 16, 0, test);
		blob * result = new blob(1,1,1,1,0,test);
		float_t m = 0;
		data->set_init_type(gaussian, 0.1);
		cacu_sumbysize_cuda(BYHEIGHT, data->s_data(), data->count(),
				1, result->s_data(), 0, 1);
		cacu_sumbysize_cpu(BYHEIGHT, data->s_data_cpu(), data->count(),
				1, result->s_data_cpu(), 0, 1);
		vec_t temp(result->count());
		result->copy2data(temp);
		float_t acc_error = 0;
		for(int i = 0; i < result->count(); ++i)
			acc_error += abs(temp[i] - result->s_data_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);
		delete data, result;
	}

	SECTION("cacu_cxsize"){
		cuda_set_device(0);
		set_rand_seed();
		blob *data = new blob(1, 32, 16, 16, 0, test);
		blob *alpha = new blob(32, 1, 1, 1, 0, test);
		blob * result = new blob(1, 32, 16, 16, 0,test);
		float_t m = 0;
		data->set_init_type(gaussian, 0.1);
		alpha->set_init_type(gaussian, 0.1);
		cacu_cxsize_cuda(data->s_data(), data->count(), alpha->s_data(),
				alpha->count(), result->s_data());
		cacu_cxsize_cpu(data->s_data_cpu(), data->count(), alpha->s_data_cpu(),
						alpha->count(), result->s_data_cpu());
		vec_t temp(result->count());
		result->copy2data(temp);
		float_t acc_error = 0;
		for(int i = 0; i < result->count(); ++i)
			acc_error += abs(temp[i] - result->s_data_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);
		delete data, result;
	}

	SECTION("cacu_cdxsize"){
		cuda_set_device(0);
		set_rand_seed();
		blob *data = new blob(1, 32, 16, 16, 0, test);
		blob *alpha = new blob(32, 1, 1, 1, 0, test);
		blob * result = new blob(1, 32, 16, 16, 0,test);
		float_t m = 0;
		data->set_init_type(gaussian, 0.1);
		alpha->set_init_type(gaussian, 0.1);
		cacu_cdxsize_cuda(data->s_data(), data->count(), alpha->s_data(),
				alpha->count(), result->s_data());
		cacu_cdxsize_cpu(data->s_data_cpu(), data->count(), alpha->s_data_cpu(),
						alpha->count(), result->s_data_cpu());
		vec_t temp(result->count());
		result->copy2data(temp);
		float_t acc_error = 0;
		for(int i = 0; i < result->count(); ++i)
			acc_error += abs(temp[i] - result->s_data_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);
		delete data, result;
	}

	SECTION("cacu_ssxpy"){
		cuda_set_device(0);
		set_rand_seed();
		blob *data = new blob(1, 32, 16, 16, 0, test);
		blob *x = new blob(32, 1, 1, 1, 0, test);
		float_t alpha = 0.1;
		float_t beta = 0.02;
		blob * result = new blob(1, 32, 16, 16, 0,test);
		float_t m = 0;
		data->set_init_type(gaussian, 0.1);
		x->set_init_type(gaussian, 0.1);
		cacu_ssxpy_cuda(x->s_data(), alpha, x->count(), result->s_data(), beta, result->count(), result->s_data());
		cacu_ssxpy_cpu(x->s_data_cpu(), alpha, x->count(), result->s_data_cpu(), beta, result->count(), result->s_data_cpu(false));
		vec_t temp(result->count());
		result->copy2data(temp);
		float_t acc_error = 0;
		for(int i = 0; i < result->count(); ++i)
			acc_error += abs(temp[i] - result->s_data_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);
		delete data, result;
	}
}

TEST_CASE("loss_functions") {
	SECTION("cacu_cross_entropy_multi"){
		cuda_set_device(0);
		set_rand_seed();
		blob *data = new blob(3, 32, 8, 8, 0, train);
		blob *result = new blob(3, 32, 8, 8, 0, train);
		bin_blob *index_ = new bin_blob(3, 1, 8, 8, 0, test);
		float_t m = 0;
		data->set_init_type(gaussian, 0.1);
		index_->set_data(2);
		cacu_cross_entropy_multi_cuda(data->s_data(), 3, 32, 8, 8, index_->s_data(), data->s_diff());
		cacu_cross_entropy_multi_cpu(data->s_data_cpu(), 3, 32, 8, 8, index_->s_data_cpu(), data->s_diff_cpu());
		vec_t temp(data->count());
		data->copy2diff(temp);
		float_t acc_error = 0;
		for(int i = 0; i < 1; ++i)
			acc_error += abs(temp[i] - data->s_diff_cpu(false)[i]);
		REQUIRE(acc_error < 1e-8);

		result->set_init_type(gaussian, 0.1);
		cacu_isaxb_cuda(result->p_data(0), data->channel(), data->width(), data->height(),
				(float_t)1, index_->p_data(0), (float_t)-1, result->p_diff(0));
		cacu_isaxb_cpu(result->p_data_cpu(0), data->channel(), data->width(), data->height(),
				(float_t)1, index_->p_data_cpu(0), (float_t)-1, result->p_diff_cpu(0));
		result->copy2diff(temp);
		float_t acc_error = 0;
		for(int i = 0; i < result->length(); ++i)
			acc_error += abs(temp[i] - result->p_diff_cpu(0, false)[i]);
		REQUIRE(acc_error < 1e-8);
		delete data, result, index_;
	}
}




#endif
#endif

#endif /* TEST_FUNCTIONS_HPP_ */
