#include "../../../mycnn.h"
#include "../../math/math_functions.h"
#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../../../3rd_party/catch.hpp"

using namespace mycnn;

TEST_CASE("convolution")
{


	SECTION("convolution data vs validate data."){
#if  __PARALLELTYPE__ == __GPU__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(1, 3, 224, 224,test);
		blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/conv/feature_map.txt");

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial convolution op
		args *args_ = new args(3,3,3,0,224,3);
		convolution_op *op = (convolution_op *)operator_factory::create_op(CACU_CONVOLUTION,input,args_);
		blob_ops::read_data2blob(op->get_weight(0),"/home/seal/cuda-workspace/CACUE/core/test/python/conv/kernel.txt");
		blob_ops::read_data2blob(op->get_weight(1),"/home/seal/cuda-workspace/CACUE/core/test/python/conv/bias.txt");

		for(int i = 0 ; i< 100; ++i)
			op->infer();

		blob *validate_ = cacu_allocator::create_blob(1, 3, 74, 74,test);
		blob_ops::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/conv/conv_result.txt");

		blob *output = op->out_data<blob>();

#if __PARALLELTYPE__ == __GPU__
		cacu_saxpby(output->s_data(),-1.0,validate_->s_data(),1.0,validate_->count());
		vec_t test(validate_->count());
		cuda_copy2host(&test[0],validate_->s_data(),test.size());
		mycnn::float_t acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		LOG_DEBUG("error : %f",acc_error);
		REQUIRE(acc_error < 0.000001);
#else
		mycnn::float_t acc_error = 0.0;
		for(int i = 0 ; i < validate_->count(); ++i){
			acc_error += abs(output->s_data()[i] - validate_->s_data()[i]);
		}
		acc_error /= validate_->count();
		LOG_DEBUG("error : %f",acc_error);
		REQUIRE(acc_error < 0.000001);
#endif

		delete op,validate_;
		delete input,args_;
#if  __PARALLELTYPE__ == __GPU__
		cublasDestroy_v2(handle);
#endif
	}
}


TEST_CASE("convolution_multidata")
{


	SECTION("convolution data vs validate data."){
#if  __PARALLELTYPE__ == __GPU__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(100, 3, 224, 224,test);
		for(int i = 0; i < 100; ++i)
			blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/conv/feature_map.txt",i);

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial convolution op
		args *args_ = new args(3,3,3,0,224,3);
		convolution_op *op = (convolution_op *)operator_factory::create_op(CACU_CONVOLUTION,input,args_);
		blob_ops::read_data2blob(op->get_weight(0),"/home/seal/cuda-workspace/CACUE/core/test/python/conv/kernel.txt");
		blob_ops::read_data2blob(op->get_weight(1),"/home/seal/cuda-workspace/CACUE/core/test/python/conv/bias.txt");

		for(int i = 0 ; i< 10; ++i)
			op->infer();

		blob *validate_ = cacu_allocator::create_blob(100, 3, 74, 74,test);
		for(int i = 0; i < 100; ++i)
			blob_ops::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/conv/conv_result.txt",i);

		blob *output = op->out_data<blob>();

#if __PARALLELTYPE__ == __GPU__
		cacu_saxpby(output->s_data(),-1.0,validate_->s_data(),1.0,validate_->count());
		vec_t test(validate_->count());
		cuda_copy2host(&test[0],validate_->s_data(),test.size());
		mycnn::float_t acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		LOG_DEBUG("error : %f",acc_error);
		REQUIRE(acc_error < 0.000001);
#else
		mycnn::float_t acc_error = 0.0;
		for(int i = 0 ; i < validate_->count(); ++i){
			acc_error += abs(output->s_data()[i] - validate_->s_data()[i]);
		}
		acc_error /= validate_->count();
		LOG_DEBUG("error : %f",acc_error);
		REQUIRE(acc_error < 0.000001);
#endif

		delete op,validate_;
		delete input,args_;
#if  __PARALLELTYPE__ == __GPU__
		cublasDestroy_v2(handle);
#endif
	}
}

TEST_CASE("convolution_grad")
{


	SECTION("convolution grad data vs validate data."){

#if  __PARALLELTYPE__ == __GPU__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(1, 3, 227, 227,train);
		blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/conv_grad/feature_map.txt");

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial convolution op
		args *args_ = new args(32,3,1,0,227,3);
		convolution_op *op = (convolution_op *)operator_factory::create_op(CACU_CONVOLUTION,input,args_);
		blob_ops::read_data2blob(op->get_weight(0),"/home/seal/cuda-workspace/CACUE/core/test/python/conv_grad/kernel.txt");

		blob *output = op->out_data<blob>();
		blob_ops::read_diff2blob(output,"/home/seal/cuda-workspace/CACUE/core/test/python/conv_grad/grad_map.txt");

		for(int i = 0 ; i < 100; ++i){
			b->_RESET_DIFF();
			op->get_weight(0)->_RESET_DIFF();
			op->get_weight(1)->_RESET_DIFF();
			op->op();
			op->grad();
		}

		blob *validate_ = cacu_allocator::create_blob(1, 3, 227, 227,test);
		blob_ops::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/conv_grad/conv_grad_result.txt");

		blob *validate_k = cacu_allocator::create_blob(32, 3, 3, 3,test);
		blob_ops::read_data2blob(validate_k,"/home/seal/cuda-workspace/CACUE/core/test/python/conv_grad/fgrad.txt");
		blob *validate_b = cacu_allocator::create_blob(32, 1, 1, 1,test);
		blob_ops::read_data2blob(validate_b,"/home/seal/cuda-workspace/CACUE/core/test/python/conv_grad/bgrad.txt");

#if __PARALLELTYPE__ == __GPU__
		cacu_saxpby(b->s_diff(),-1.0,validate_->s_data(),1.0,validate_->count());
		vec_t test(validate_->count());
		cuda_copy2host(&test[0],validate_->s_data(),test.size());
		mycnn::float_t acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		LOG_DEBUG("gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);

		cacu_saxpby(op->get_weight(0)->s_diff(),-1.0,validate_k->s_data(),1.0,validate_k->count());
		test.resize(validate_k->count());
		cuda_copy2host(&test[0],validate_k->s_data(),test.size());
		acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		LOG_DEBUG("w gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);

		cacu_saxpby(op->get_weight(1)->s_diff(),-1.0,validate_b->s_data(),1.0,validate_b->count());
		test.resize(validate_b->count());
		cuda_copy2host(&test[0],validate_b->s_data(),test.size());
		acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		LOG_DEBUG("bias gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.0001);
#else
		mycnn::float_t acc_error = 0.0;
		for(int i = 0 ; i < validate_->count(); ++i){
			acc_error += abs(b->s_diff()[i] - validate_->s_data()[i]);
		}
		acc_error /= validate_->count();
		LOG_DEBUG("gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);

		acc_error = 0.0;
		for(int i = 0 ; i < validate_k->count(); ++i){
			acc_error += abs(op->get_weight(0)->s_diff()[i] - validate_k->s_data()[i]);
		}
		acc_error /= validate_k->count();
		LOG_DEBUG("w gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);

		acc_error = 0.0;
		for(int i = 0 ; i < validate_b->count(); ++i){
			acc_error += abs(op->get_weight(1)->s_diff()[i] - validate_b->s_data()[i]);
		}
		acc_error /= validate_b->count();
		LOG_DEBUG("bias gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);
#endif

		delete op,validate_,validate_k,validate_b;
		delete input,args_;
#if  __PARALLELTYPE__ == __GPU__
		cublasDestroy_v2(handle);
#endif
	}
}

TEST_CASE("convolution_grad_multidata")
{


	SECTION("convolution grad data vs validate data."){

#if  __PARALLELTYPE__ == __GPU__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(100, 3, 227, 227,train);
		for(int i = 0 ; i< 100; ++i)
			blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/conv_grad/feature_map.txt",i);

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial convolution op
		args *args_ = new args(32,3,1,0,227,3);
		convolution_op *op = (convolution_op *)operator_factory::create_op(CACU_CONVOLUTION,input,args_);
		blob_ops::read_data2blob(op->get_weight(0),"/home/seal/cuda-workspace/CACUE/core/test/python/conv_grad/kernel.txt");

		blob *output = op->out_data<blob>();
		for(int i = 0 ; i< 100; ++i)
			blob_ops::read_diff2blob(output,"/home/seal/cuda-workspace/CACUE/core/test/python/conv_grad/grad_map.txt",i);

		for(int i = 0 ; i < 10; ++i){
			b->_RESET_DIFF();
			op->get_weight(0)->_RESET_DIFF();
			op->get_weight(1)->_RESET_DIFF();
			op->op();
			op->grad();
		}

		blob *validate_ = cacu_allocator::create_blob(100, 3, 227, 227,test);
		for(int i = 0; i < 100; ++i)
			blob_ops::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/conv_grad/conv_grad_result.txt",i);

		blob *validate_k = cacu_allocator::create_blob(32, 3, 3, 3,test);
		blob_ops::read_data2blob(validate_k,"/home/seal/cuda-workspace/CACUE/core/test/python/conv_grad/fgrad.txt");
		blob *validate_b = cacu_allocator::create_blob(32, 1, 1, 1,test);
		blob_ops::read_data2blob(validate_b,"/home/seal/cuda-workspace/CACUE/core/test/python/conv_grad/bgrad.txt");

#if __PARALLELTYPE__ == __GPU__
		cacu_saxpby(b->s_diff(),-1.0,validate_->s_data(),1.0,validate_->count());
		vec_t test(validate_->count());
		cuda_copy2host(&test[0],validate_->s_data(),test.size());
		mycnn::float_t acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		LOG_DEBUG("gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);

		cacu_scalex(op->get_weight(0)->s_diff(),op->get_weight(0)->count(),mycnn::float_t(1)/100);
		cacu_saxpby(op->get_weight(0)->s_diff(),-1.0,validate_k->s_data(),1.0,validate_k->count());
		test.resize(validate_k->count());
		cuda_copy2host(&test[0],validate_k->s_data(),test.size());
		acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		LOG_DEBUG("w gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);

		cacu_scalex(op->get_weight(1)->s_diff(),op->get_weight(1)->count(),mycnn::float_t(1)/100);
		cacu_saxpby(op->get_weight(1)->s_diff(),-1.0,validate_b->s_data(),1.0,validate_b->count());
		test.resize(validate_b->count());
		cuda_copy2host(&test[0],validate_b->s_data(),test.size());
		acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		LOG_DEBUG("bias gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.0001);
#else
		mycnn::float_t acc_error = 0.0;
		for(int i = 0 ; i < validate_->count(); ++i){
			acc_error += abs(b->s_diff()[i] - validate_->s_data()[i]);
		}
		acc_error /= validate_->count();
		LOG_DEBUG("gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);

		acc_error = 0.0;
		cacu_scalex(op->get_weight(0)->s_diff(),op->get_weight(0)->count(), (mycnn::float_t)1/100);
		for(int i = 0 ; i < validate_k->count(); ++i){
			acc_error += abs(op->get_weight(0)->s_diff()[i] - validate_k->s_data()[i]);
		}
		acc_error /= validate_k->count();
		LOG_DEBUG("w gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.0001);

		acc_error = 0.0;
		cacu_scalex(op->get_weight(1)->s_diff(),op->get_weight(1)->count(), (mycnn::float_t)1/100);
		for(int i = 0 ; i < validate_b->count(); ++i){
			acc_error += abs(op->get_weight(1)->s_diff()[i] - validate_b->s_data()[i]);
		}
		acc_error /= validate_b->count();
		LOG_DEBUG("bias gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);
#endif

		delete op,validate_,validate_k,validate_b;
		delete input,args_;
#if  __PARALLELTYPE__ == __GPU__
		cublasDestroy_v2(handle);
#endif
	}
}
