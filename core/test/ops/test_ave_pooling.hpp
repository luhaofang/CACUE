#include "../../../mycnn.h"
#include "../../math/math_functions.h"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../../../3rd_party/catch.hpp"

using namespace mycnn;

TEST_CASE("average_pooling")
{


	SECTION("ave_pooling data vs validate data."){
#if  __PARALLELTYPE__ == __CUDA__
		cublasCreate_v2(&handle);
#endif
		blob *b = cacu_allocator::create_blob(1, 3, 224, 224, test);
		blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/pooling/a_feature_map.txt");

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial convolution op
		args *args_ = new args(3,3,1,0,0,0);
		average_pooling_op *op = (average_pooling_op *)operator_factory::create_op(CACU_AVERAGE_POOLING,input,args_);

		for(int i =0;i < 1000;++i)
			op->infer();

		blob *validate_ = cacu_allocator::create_blob(1, 3, 222, 222, test);
		blob_ops::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/pooling/a_result.txt");

		blob *output = op->out_data<blob>();

#if __PARALLELTYPE__ == __CUDA__
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
