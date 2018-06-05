#include "test/test_cases.h"

//#include "example/cifar10/train.hpp"
//#include "example/cifar10/test.hpp"
//#include "example/cifar10/data_proc.h"
//#include "example/mtcnn/train_pnet.hpp"
//#include "example/mtcnn/demo.hpp"

//#include "example/frcnn/train_modified.hpp"
//#include "example/imagenet/image_test.hpp"
//#include "example/frcnn/test.hpp"

//#include "example/imagenet/train_modified.hpp"
#include "example/generate/train_cifar.hpp"
//#include "example/generate/test.hpp"



int main(int argc, char** argv) {

	train_net();
	//test_net();
	//int result =  Catch::Session().run(argc,argv);
	//make_mean("/home/haofang/data/cifar10/","/home/haofang/data/cifar10/mean.binproto");
}

