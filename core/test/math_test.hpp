
#include <time.h>
#include "../math/math_function_openmp.h"

#include "../../mycnn.h"

using namespace mycnn;


void test_math()
{
	//printf("fuck");
//	blob *b = cacu_allocator::create_blob(1, 1, 5, 5, 1 , train);
//	layer *layer_ = new layer(1, 3, 1, 1);
//
//	layer_->op(CACU_BATCH_NORMALIZE, b)->op(CACU_RELU);
//	layer_->operate();
	//average_pooling_op* op = layer_->get_op<average_pooling_op>(0);


	weight *_b = new weight("test",1, 1, 5, 3,train);
	_b->set_init_type(gaussian,1);
	cacu_print(_b->s_data(),_b->count());
	cacu_transpose(_b->s_data(),5,3);
	cacu_print(_b->s_data(),_b->count());
	delete _b;
}
