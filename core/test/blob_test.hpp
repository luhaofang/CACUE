#include "../../mycnn.h"
#include <time.h>

using namespace mycnn;


void test_blob()
{
	blob * b = cacu_allocator::create_blob(1,3,224,224,test);
	blob_tools::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/feature_map.txt");
}
