#include "mycnn.h"
#include <time.h>

#include "core/test/blob_test.hpp"
#include "core/test/op_test.hpp"
#include "core/test/network_test.hpp"
#include "core/test/graph_test.hpp"
#include "core/test/math_test.hpp"
#include "core/test/sovler_test.hpp"
#include "core/test/iotools_test.hpp"

//#include "example/cifar10/train.hpp"
//#include "example/cifar10/test.hpp"
//#include "example/cifar10/dy_train.hpp"
//#include "example/cifar10/image_test.hpp"

//#include "example/imagenet/train.hpp"
//#include "example/imagenet/train_modified.hpp"
//#include "example/imagenet/asyn_train.hpp"
//#include "example/imagenet/dy_train.hpp"
#include "example/imagenet/test.hpp"
//#include "example/imagenet/image_test.hpp"

//#include "example/cifar10/data_proc.hpp"
//#include "example/imagenet/data_proc.hpp"

#include "core/test/test_cases.h"

using namespace mycnn;
using namespace mycnn_tools;


int main(int argc, char** argv) {
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
	//string datapath = "/home/seal/4T/imagenet/224X224_train/";
	//string filelist = "/home/seal/4T/imagenet/file_list.txt";
	//string meanpath = "/home/seal/4T/imagenet/224X224_mean.binproto";
	//make_mean(datapath,filelist,meanpath);
	//train_net();

	test_net();

	//int result =  Catch::Session().run(argc,argv);

}

