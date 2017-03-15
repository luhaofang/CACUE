#include "../../../mycnn.h"
#include "../../math/math_functions.h"
#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../../../3rd_party/catch.hpp"

using namespace mycnn;

TEST_CASE("convolution")
{


	SECTION("."){
#if  __PARALLELTYPE__ == __GPU__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(1, 3, 224, 224,test);
		blob_tools::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/conv/feature_map.txt");

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial convolution op
		args *args_ = new args(3,3,3,0,224,3);
		convolution_op *op = (convolution_op *)operator_factory::create_op(CACU_CONVOLUTION,input,args_);
		blob_tools::read_data2blob(op->get_weight(0),"/home/seal/cuda-workspace/CACUE/core/test/python/conv/kernel.txt");
		blob_tools::read_data2blob(op->get_weight(1),"/home/seal/cuda-workspace/CACUE/core/test/python/conv/bias.txt");

		op->infer();

		blob *validate_ = cacu_allocator::create_blob(1, 3, 74, 74,test);
		blob_tools::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/conv/conv_result.txt");

		blob *output = (blob*)op->out_data();

#if __PARALLELTYPE__ == __GPU__
		cacu_saxpby(output->s_data(),-1.0,validate_->s_data(),1.0,validate_->count());
		vec_t test(validate_->count());
		cuda_copy2host(&test[0],validate_->s_data(),test.size());
		float_t acc_error = 0.0;
		for(int i = 0 ; i < test.size(); ++i){
			acc_error += abs(test[i]);
		}
		acc_error /= test.size();
		REQUIRE(acc_error < 0.000001);
#else
		float_t acc_error = 0.0;
		for(int i = 0 ; i < validate_->count(); ++i){
			acc_error += abs(output->s_data()[i] - validate_->s_data()[i]);
		}
		acc_error /= validate_->count();
		REQUIRE(acc_error < 0.000001);
#endif

		delete op,validate_;
		delete input,args_;
#if  __PARALLELTYPE__ == __GPU__
		cublasDestroy_v2(handle);
#endif
	}
}

