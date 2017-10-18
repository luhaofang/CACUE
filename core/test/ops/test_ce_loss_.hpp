#include "../../../mycnn.h"
#include "../../math/math_functions.h"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../../../3rd_party/catch.hpp"

using namespace mycnn;

TEST_CASE("cross_entropy")
{


	SECTION("cross_entropy data vs validate data."){
#if  __PARALLELTYPE__ == __CUDA__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(1, 1000, 1, 1, test);
		blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/loss/feature_map.txt");

		bin_blob *label = cacu_allocator::create_bin_blob(1, 1, 1, 1, 1, test);

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);
		input->push_back(label);

		//initial convolution op
		args *args_ = new args(1000,2,2,0,0,0);
		softmax_with_loss_op *op = (softmax_with_loss_op *)operator_factory::create_op(CACU_SOFTMAX_LOSS,input,args_);

		for(int i =0;i < 100;++i)
			op->infer();
		op->echo();

		//blob *validate_ = cacu_allocator::create_blob(1, 64, 112, 112, test);
		//blob_ops::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/pooling/conv_result.txt");

		blob *output = op->out_data<blob>();

		delete op;
		delete input,args_;
#if  __PARALLELTYPE__ == __CUDA__
		cublasDestroy_v2(handle);
#endif
	}
}


TEST_CASE("cross_entropy_grad")
{

	SECTION("cross_entropy grad data vs validate data."){

#if  __PARALLELTYPE__ == __CUDA__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(1, 1000, 1, 1,train);
		blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/loss/feature_map.txt");

		bin_blob *label = cacu_allocator::create_bin_blob(1, 1, 1, 1, 1, test);

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);
		input->push_back(label);

		//initial average_pooing op
		args *args_ = new args(1000,2,2,0,224,64);
		softmax_with_loss_op *op = (softmax_with_loss_op *)operator_factory::create_op(CACU_SOFTMAX_LOSS,input,args_);

		//blob *output = op->out_data<blob>();
		//blob_ops::read_diff2blob(output,"/home/seal/cuda-workspace/CACUE/core/test/python/pooling/grad_map.txt");

		for(int i = 0 ; i < 100; ++i){
			b->_RESET_DIFF();
			op->op();
			op->grad();
		}

		blob *validate_ = cacu_allocator::create_blob(1, 1000, 1, 1,test);
		blob_ops::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/loss/conv_grad_result.txt");

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

