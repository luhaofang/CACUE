#include "../../../mycnn.h"
#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../../../3rd_party/catch.hpp"

using namespace mycnn;

TEST_CASE("convolution")
{


	SECTION("convolution data vs validate data."){

		blob *b = cacu_allocator::create_blob(1, 3, 224, 224,test);
		blob_tools::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/feature_map.txt");

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial convolution op
		args *args_ = new args(3,3,1,0,224,3);
		convolution_op *op = (convolution_op *)operator_factory::create_op(CACU_CONVOLUTION,input,args_);
		blob_tools::read_data2blob(op->get_weight(0),"/home/seal/cuda-workspace/CACUE/core/test/python/kernel.txt");
		blob_tools::read_data2blob(op->get_weight(1),"/home/seal/cuda-workspace/CACUE/core/test/python/bias.txt");

		op->infer();

		blob *validate_ = cacu_allocator::create_blob(1, 3, 222, 222,test);
		blob_tools::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/conv_result.txt");

		blob *output = (blob*)op->out_data();
		float_t acc_error = 0.0;
		for(int i = 0 ; i < 3*222*222; ++i){
			acc_error += abs(output->s_data()[i] - validate_->s_data()[i]);
		}
		acc_error /= validate_->count();
		REQUIRE(acc_error < 0.000001);

		delete op,validate_;
		delete input,args_;
	}
}

TEST_CASE("convolution_grad")
{


	SECTION("convolution data vs validate data."){

		blob *b = cacu_allocator::create_blob(1, 3, 224, 224,test);
		blob_tools::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/feature_map.txt");

		blobs *input = cacu_allocator::create_blobs();
		input->push_back(b);

		//initial convolution op
		args *args_ = new args(3,3,1,0,224,3);
		convolution_op *op = (convolution_op *)operator_factory::create_op(CACU_CONVOLUTION,input,args_);
		blob_tools::read_data2blob(op->get_weight(0),"/home/seal/cuda-workspace/CACUE/core/test/python/kernel.txt");
		blob_tools::read_data2blob(op->get_weight(1),"/home/seal/cuda-workspace/CACUE/core/test/python/bias.txt");

		op->infer();

		blob *validate_ = cacu_allocator::create_blob(1, 3, 222, 222,test);
		blob_tools::read_data2blob(validate_,"/home/seal/cuda-workspace/CACUE/core/test/python/conv_result.txt");

		blob *output = (blob*)op->out_data();
		float_t acc_error = 0.0;
		for(int i = 0 ; i < 3*222*222; ++i){
			acc_error += abs(output->s_data()[i] - validate_->s_data()[i]);
		}
		acc_error /= validate_->count();
		REQUIRE(acc_error < 0.000001);

		delete op,validate_;
		delete input,args_;
	}
}
