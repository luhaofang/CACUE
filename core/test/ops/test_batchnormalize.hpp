#include "../../../mycnn.h"
#include "../../math/math_functions.h"
#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../../../3rd_party/catch.hpp"

#include "../../utils/blob_ops.h"

using namespace mycnn;

TEST_CASE("batch_normalize")
{


	SECTION("ip data vs validate data."){
#if  __PARALLELTYPE__ == __CUDA__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(1, 3, 224, 224,train);
		blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/feature_map.txt");

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial convolution op
		args *args_ = new args(3,0,0,0,0,0);
		batch_normalize_op *op = (batch_normalize_op *)operator_factory::create_op(CACU_BATCH_NORMALIZE,input,args_);
		blob_ops::read_data2blob(op->get_weight(0),"/home/seal/cuda-workspace/CACUE/core/test/python/bn/kernel.txt");
		blob_ops::read_data2blob(op->get_weight(1),"/home/seal/cuda-workspace/CACUE/core/test/python/bn/bias.txt");

		for(int i = 0 ; i<10; ++i)
			op->infer();

		blob *validate_ = cacu_allocator::create_blob(1, 3, 224, 224,test);
		blob_ops::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/conv_result.txt");

		blob *output = op->out_data<blob>();

#if __PARALLELTYPE__ == __CUDA__
		//cacu_print(output->s_data(),100);
		//cacu_print(validate_->s_data(),100);
		cacu_saxpby(output->s_data(),(mycnn::float_t)-1.0,validate_->s_data(),(mycnn::float_t)1.0,validate_->count());
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
			//printf("%d:(%f,%f),",i,output->s_data()[i],validate_->s_data()[i]);
		}
		acc_error /= validate_->count();
		LOG_DEBUG("error : %f",acc_error);
		REQUIRE(acc_error < 0.0001);
#endif

		delete op,validate_;
		delete input,args_;
#if  __PARALLELTYPE__ == __CUDA__
		cublasDestroy_v2(handle);
#endif
	}
}


TEST_CASE("batch_normalize_multidata")
{


	SECTION("ip data vs validate data."){
#if  __PARALLELTYPE__ == __CUDA__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(100, 3, 224, 224,train);
		for(int i = 0 ; i< 100; ++i)
			blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/feature_map.txt",i);

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial convolution op
		args *args_ = new args(3,0,0,0,0,0);
		batch_normalize_op *op = (batch_normalize_op *)operator_factory::create_op(CACU_BATCH_NORMALIZE,input,args_);
		blob_ops::read_data2blob(op->get_weight(0),"/home/seal/cuda-workspace/CACUE/core/test/python/bn/kernel.txt");
		blob_ops::read_data2blob(op->get_weight(1),"/home/seal/cuda-workspace/CACUE/core/test/python/bn/bias.txt");

		for(int i = 0 ; i<10; ++i)
			op->infer();

		blob *validate_ = cacu_allocator::create_blob(100, 3, 224, 224,test);
		for(int i = 0 ; i< 100; ++i)
			blob_ops::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/conv_result.txt",i);

		blob *output = op->out_data<blob>();

#if __PARALLELTYPE__ == __CUDA__
		cacu_saxpby(output->s_data(),(mycnn::float_t)-1.0,validate_->s_data(),(mycnn::float_t)1.0,validate_->count());
		vec_t test(validate_->count());
		cuda_copy2host(&test[0],validate_->s_data(),test.size());
		mycnn::float_t acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		LOG_DEBUG("error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);
#else
		mycnn::float_t acc_error = 0.0;
		for(int i = 0 ; i < validate_->count(); ++i){
			acc_error += abs(output->s_data()[i] - validate_->s_data()[i]);
			//printf("%d:(%f,%f),",i,output->s_data()[i],validate_->s_data()[i]);
		}
		acc_error /= validate_->count();
		LOG_DEBUG("error : %f",acc_error);
		REQUIRE(acc_error < 0.0001);
#endif

		delete op,validate_;
		delete input,args_;
#if  __PARALLELTYPE__ == __CUDA__
		cublasDestroy_v2(handle);
#endif
	}
}

TEST_CASE("batch_normalize_grad")
{


	SECTION("ip grad data vs validate data."){

#if  __PARALLELTYPE__ == __CUDA__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(1, 3, 224, 224,train);
		blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/feature_map.txt");

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial convolution op
		args *args_ = new args(3,0,0,0,0,0);
		batch_normalize_op *op = (batch_normalize_op *)operator_factory::create_op(CACU_BATCH_NORMALIZE,input,args_);
		blob_ops::read_data2blob(op->get_weight(0),"/home/seal/cuda-workspace/CACUE/core/test/python/bn/kernel.txt");
		blob_ops::read_data2blob(op->get_weight(1),"/home/seal/cuda-workspace/CACUE/core/test/python/bn/bias.txt");

		blob *output = op->out_data<blob>();
		blob_ops::read_diff2blob(output,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/grad_map.txt");

		for(int i = 0 ; i < 10; ++i){
			b->_RESET_DIFF();
			op->get_weight(0)->_RESET_DIFF();
			op->get_weight(1)->_RESET_DIFF();
			op->op();
			op->grad();
		}

		blob *validate_ = cacu_allocator::create_blob(1, 3, 224, 224,test);
		blob_ops::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/conv_grad_result.txt");

		blob *validate_k = cacu_allocator::create_blob(3, 1, 1, 1,test);
		blob_ops::read_data2blob(validate_k,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/fgrad.txt");
		blob *validate_b = cacu_allocator::create_blob(3, 1, 1, 1,test);
		blob_ops::read_data2blob(validate_b,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/bgrad.txt");

#if __PARALLELTYPE__ == __CUDA__

		cacu_saxpby(b->s_diff(),(mycnn::float_t)-1.0,validate_->s_data(),(mycnn::float_t)1.0,validate_->count());
		vec_t test(validate_->count());
		cuda_copy2host(&test[0],validate_->s_data(),test.size());
		mycnn::float_t acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		LOG_DEBUG("gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.0001);

		cacu_saxpby(op->get_weight(0)->s_diff(),(mycnn::float_t)-1.0,validate_k->s_data(),(mycnn::float_t)1.0,validate_k->count());
		test.resize(validate_k->count());
		cuda_copy2host(&test[0],validate_k->s_data(),test.size());
		acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		LOG_DEBUG("w gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);

		cacu_saxpby(op->get_weight(1)->s_diff(),(mycnn::float_t)-1.0,validate_b->s_data(),(mycnn::float_t)1.0,validate_b->count());
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
#if  __PARALLELTYPE__ == __CUDA__
		cublasDestroy_v2(handle);
#endif
	}
}


TEST_CASE("batch_normalize_grad_multidata")
{


	SECTION("ip grad data vs validate data."){

#if  __PARALLELTYPE__ == __CUDA__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(100, 3, 224, 224,train);
		for(int i = 0 ; i< 100; ++i)
			blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/feature_map.txt",i);

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial convolution op
		args *args_ = new args(3,0,0,0,0,0);
		batch_normalize_op *op = (batch_normalize_op *)operator_factory::create_op(CACU_BATCH_NORMALIZE,input,args_);
		blob_ops::read_data2blob(op->get_weight(0),"/home/seal/cuda-workspace/CACUE/core/test/python/bn/kernel.txt");
		blob_ops::read_data2blob(op->get_weight(1),"/home/seal/cuda-workspace/CACUE/core/test/python/bn/bias.txt");

		blob *output = op->out_data<blob>();
		for(int i = 0 ; i< 100; ++i)
			blob_ops::read_diff2blob(output,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/grad_map.txt",i);

		for(int i = 0 ; i < 1; ++i){
			b->_RESET_DIFF();
			op->get_weight(0)->_RESET_DIFF();
			op->get_weight(1)->_RESET_DIFF();
			op->op();
			op->grad();
		}

		blob *validate_ = cacu_allocator::create_blob(100, 3, 224, 224,test);
		for(int i = 0 ; i< 100; ++i)
			blob_ops::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/conv_grad_result.txt",i);

		blob *validate_k = cacu_allocator::create_blob(3, 1, 1, 1,test);
		blob_ops::read_data2blob(validate_k,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/fgrad.txt");
		blob *validate_b = cacu_allocator::create_blob(3, 1, 1, 1,test);
		blob_ops::read_data2blob(validate_b,"/home/seal/cuda-workspace/CACUE/core/test/python/bn/bgrad.txt");

#if __PARALLELTYPE__ == __CUDA__

		cacu_saxpby(b->s_diff(),(mycnn::float_t)-1.0,validate_->s_data(),(mycnn::float_t)1.0,validate_->count());
		vec_t test(validate_->count());
		cuda_copy2host(&test[0],validate_->s_data(),test.size());
		mycnn::float_t acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		LOG_DEBUG("gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.0001);

		cacu_scalex(op->get_weight(0)->s_diff(),op->get_weight(0)->count(), (mycnn::float_t)1/100);
		cacu_saxpby(op->get_weight(0)->s_diff(),(mycnn::float_t)-1.0,validate_k->s_data(),(mycnn::float_t)1.0,validate_k->count());
		test.resize(validate_k->count());
		cuda_copy2host(&test[0],validate_k->s_data(),test.size());
		acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		LOG_DEBUG("w gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);

		cacu_scalex(op->get_weight(1)->s_diff(),op->get_weight(1)->count(), (mycnn::float_t)1/100);
		cacu_saxpby(op->get_weight(1)->s_diff(),(mycnn::float_t)-1.0,validate_b->s_data(),(mycnn::float_t)1.0,validate_b->count());
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

		cacu_scalex(op->get_weight(0)->s_diff(),op->get_weight(0)->count(), (mycnn::float_t)1/100);
		acc_error = 0.0;
		for(int i = 0 ; i < validate_k->count(); ++i){
			acc_error += abs(op->get_weight(0)->s_diff()[i] - validate_k->s_data()[i]);
		}
		acc_error /= validate_k->count();
		LOG_DEBUG("w gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);

		cacu_scalex(op->get_weight(1)->s_diff(),op->get_weight(1)->count(), (mycnn::float_t)1/100);
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
#if  __PARALLELTYPE__ == __CUDA__
		cublasDestroy_v2(handle);
#endif
	}
}

