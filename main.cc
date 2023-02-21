#include "test/test_cases.h"
//#include "cacu/cacu.h"

//#include "example/cifar10/train.hpp"
//#include "example/cifar10/test.hpp"
//#include "example/cifar10/for_one_test.hpp"
#include "example/cifar10/for_resnet_test.hpp"
//#include "example/cifar10/train_modified.hpp"
//#include "example/cifar10/train_modified_multiple.hpp"
//#include "example/cifar10/test_modified_multiple.hpp"
//#include "example/cifar10/data_proc.h"
//#include "example/mtcnn/train_pnet.hpp"
//#include "example/mtcnn/demo.hpp"

//#include "example/mnist/train.hpp"
//#include "example/mnist/test.hpp"
//#include "example/mnist/mnist_data_proc.h"

//#include "example/frcnn/train_modified.hpp"
//#include "example/imagenet/image_test.hpp"
//#include "example/frcnn/test.hpp"

//#include "example/imagenet/train_modified.hpp"
//#include "example/imagenet/test.hpp"
//#include "example/imagenet/performance.hpp"
//#include "example/gan/train_cifar.hpp"
//#include "example/gan/train_modified_96.hpp"
//#include "example/generate/test.hpp"
//#include "example/cgan/train.hpp"

//#include "example/cycle_gan/train_conditional.hpp"
//#include "example/cycle_gan/train.hpp"
//#include "example/cycle_gan/train_conditional_nocycle.hpp"
//#include "example/super_resolution/train.hpp"

int main(int argc, char** argv) {
	//CACUE_PRINTER();
	//train_net();
	//make_mean("/Users/seallhf/Documents/datasets/cifar10/", "/Users/seallhf/Documents/datasets/cifar10/mean.binproto");
	test_net();
//	int result =  Catch::Session().run(argc,argv);
//	for (int i = 1; i <= 5; i++)
//	{
//		ostringstream oss;
//		oss << "/Users/seallhf/Documents/datasets/cifar10/" << "data_batch_" << i << ".bin";//"test_batch.bin";
//		output_224x224((oss.str()), 1, "/Users/seallhf/Documents/datasets/cifar10/224x224/train/", 224, 224);
//	}
	//make_mean_mnist("/Users/seallhf/Documents/mywork/data/mnist/","/Users/seallhf/Documents/mywork/data/mnist/mean.binproto");
}

