#include "../../mycnn.h"
#include <time.h>

#include "../utils/blob_ops.h"

using namespace mycnn;


void test_blob()
{
	blob * b = cacu_allocator::create_blob(1,3,224,224,test);
	blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/feature_map.txt");
}
