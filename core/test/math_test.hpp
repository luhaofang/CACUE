
#include <time.h>
#include "../math/math_function_openmp.h"

#include "../../mycnn.h"

using namespace mycnn;


void test_math()
{

	blob *b = cacu_allocator::create_blob(1, 256, 14, 14, 1 , train);
	//layer *layer_ = new layer(512, 3, 1, 1);


	//layer_->op(CACU_CONVOLUTION, b)->op(CACU_RELU);
	//average_pooling_op* op = layer_->get_op<average_pooling_op>(0);


	network *net = create_alexnet();

	clock_t start = clock();
	net->predict();
	clock_t end = clock();
	LOG_INFO("time costs:%d",end - start);
}
