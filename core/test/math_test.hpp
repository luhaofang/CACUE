
#include <time.h>
#include "../math/math_function_openmp.h"

#include "../../mycnn.h"

using namespace mycnn;


void test_math()
{
	//printf("fuck");
	blob *b = cacu_allocator::create_blob(1, 1, 5, 5, 1 , train);
	layer *layer_ = new layer(1, 3, 1, 1);

	layer_->op(CACU_BATCH_NORMALIZE, b)->op(CACU_RELU);
	layer_->operate();
	//average_pooling_op* op = layer_->get_op<average_pooling_op>(0);


//	network *net = create_alexnet();
//
//	clock_t start = clock();
//	net->predict();
//	clock_t end = clock();
//	LOG_INFO("time costs:%d",end - start);

}
