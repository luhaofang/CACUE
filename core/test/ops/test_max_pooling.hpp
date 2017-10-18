#include "../../../mycnn.h"
#include "../../math/math_functions.h"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../../../3rd_party/catch.hpp"

using namespace mycnn;

TEST_CASE("max_pooling")
{


	SECTION("ave_pooling data vs validate data."){
#if  __PARALLELTYPE__ == __CUDA__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(1, 64, 224, 224, test);
		blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/pooling/m_feature_map.txt");

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial convolution op
		args *args_ = new args(64,2,2,0,0,0);
		max_pooling_op *op = (max_pooling_op *)operator_factory::create_op(CACU_MAX_POOLING,input,args_);

		for(int i =0;i < 1;++i)
			op->infer();

		blob *validate_ = cacu_allocator::create_blob(1, 64, 112, 112, test);
		blob_ops::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/pooling/m_conv_result.txt");

		blob *output = op->out_data<blob>();

#if __PARALLELTYPE__ == __CUDA__
		//cacu_print(output->s_data(),1000);
		//cacu_print(validate_->s_data(),1000);
		cacu_saxpby(output->s_data(),-1.0,validate_->s_data(),1.0,validate_->count());
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
		REQUIRE(acc_error < 0.00001);
#endif

		delete op,validate_;
		delete input,args_;
#if  __PARALLELTYPE__ == __CUDA__
		cublasDestroy_v2(handle);
#endif
	}
}


TEST_CASE("max_pooling_grad")
{


	SECTION("pooling grad data vs validate data."){

#if  __PARALLELTYPE__ == __CUDA__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(1, 64, 224, 224,train);
		blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/pooling/m_feature_map.txt");

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial max_pooing op
		args *args_ = new args(64,2,2,0,224,64);
		max_pooling_op *op = (max_pooling_op *)operator_factory::create_op(CACU_MAX_POOLING,input,args_);

		blob *output = op->out_data<blob>();
		blob_ops::read_diff2blob(output,"/home/seal/cuda-workspace/CACUE/core/test/python/pooling/m_grad_map.txt");

		for(int i = 0 ; i < 1; ++i){
			b->_RESET_DIFF();
			op->op();
			op->grad();
		}

		blob *validate_ = cacu_allocator::create_blob(1, 64, 224, 224,test);
		blob_ops::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/pooling/m_conv_grad_result.txt");

#if __PARALLELTYPE__ == __CUDA__

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

#else
		mycnn::float_t acc_error = 0.0;
		for(int i = 0 ; i < validate_->count(); ++i){
			acc_error += abs(b->s_diff()[i] - validate_->s_data()[i]);
		}
		acc_error /= validate_->count();
		LOG_DEBUG("gradient error : %f",acc_error);
		REQUIRE(acc_error < 0.00001);

#endif

		delete op,validate_;
		delete input,args_;
#if  __PARALLELTYPE__ == __CUDA__
		cublasDestroy_v2(handle);
#endif
	}
}

