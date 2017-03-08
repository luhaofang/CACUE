#include "mycnn.h"
#include <time.h>

#include "core/test/blob_test.hpp"
#include "core/test/op_test.hpp"
#include "core/test/network_test.hpp"
#include "core/test/graph_test.hpp"
#include "core/test/math_test.hpp"
#include "core/test/sovler_test.hpp"
#include "core/test/iotools_test.hpp"

#include "example/cifar10/train.hpp"

using namespace mycnn;
using namespace mycnn_tools;

int main()
{
	//test_blob();
	//test_op();
	//test_network();
	//test_graph();
	//test_math();
	//test_sovler();
	//test_iotools();
	//string datapath = "/home/seal/4T/cacue/cifar10/data/";
	//string meanpath = "/home/seal/4T/cacue/cifar10/data/mean.binproto";
	//make_mean(datapath,meanpath);
	train_net();

}
