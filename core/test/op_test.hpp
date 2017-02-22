#include "../../mycnn.h"
#include <time.h>

using namespace mycnn;

void test_op()
{

	blob *b = cacu_allocator::create_blob(1, 1, 1, 1,test);
	/*inner_product_op *c = new inner_product_op(b,new ARGS());
	weight<mycnn::float_t> *blobs = new weight<mycnn::float_t>("a", 1, 1, 1, 1);

	
	c->set_weight(blobs);*/
}
