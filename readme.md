# CACU's Evolution version [![Build Status](https://travis-ci.org/luhaofang/CACUE.svg?branch=master)](https://travis-ci.org/luhaofang/CACUE)

CACUE is a light weighted Deep learning framework based on standard C++11. Aimed at the engineering aspect usage of deep learning projects.
Contains different kinds of released models, includes classification models: 'lenet','vgg16','res18','res50','mobilenet', face detection: 'mtcnn', GANS: 'DCGAN' on cifar10, 'CycleGAN', etc. The framework is written by David Lu.

## Brief

We intent to create an easily read and introduced DNN framework. By using the sample logic code, you can complie your DNN model on different kinds of devices. CACUE don't have many definitions, we've decoupled the operator algorithm logic from mathmetic calculation. You just need to focus on the operator compute logic, once you want to create new compute operator. By setting differnet definition, CACUE could help you to fast compute on different device. Also CACUE supports both dynamic computing and static computing. You may find  that CACUE's operator could be used as differentiable operator, we also supply different mathmetic operators.

## Features

- Easily included in your system.

 		#include "cacu.h" 
		using namespace cacu;
  that's all you need to do. If you want to compile with blas, open ROOT_PATH/config.h.
  
  	#define __CBLASTYPE__ __OPENBLAS__ // for cblas usage.
  	#define __PARALLELTYPE__  __OPENBLAS__ // for parallel type usage.
  
  You can set use_deivce on, if you want to use GPU or other avaible computing deivce to compile CACUE.
  Less dependencies(opencv,openblas,mkl,cuda.cudnn) or NO dependency, that's all depends on your project.
  
- Switch on static computing and dynamic computing.

		#define __OPERATOR__TYPE__ __DYNAMIC_GRAPH__

		cacu_op *conv = new cacu_op(CACU_CONVOLUTION, new data_args(32, 32, 3, 3, 3), train);
		conv->get_param(0)->set_init_type(gaussian,0.1);
		conv->forward(blobs);
		
  Dynamic computing is and important feature for a lot of algorithm but not in all scenes, CACUE provide easily method for the change.
  It's a flexiable usage for operator using.
  
- Support unified math logic functions.
  DON'T need to focus on the heterogeneous environment. All operator just need to implement the operator logic.
  
## Examples

We provide some of the example models that trained based on CACUE.

### mnist && cifar10

create mean file:

    #include "example/mnist/mnist_data_proc.h"
    //generate mean data
    make_mean_mnist("/path/to/mnist/data/", "/path/to/mean.data");

train mnist model (cifar10 almost the same.):

    //train model
    #include <time.h>

    #include "../../cacu/solvers/sgd_solver.h"
    #include "../../cacu/solvers/adam_solver.h"
    
    #include "../../cacu/cacu.h"
    #include "../../cacu/config.h"
    
    #include "../../tools/imageio_utils.h"
    #include "../../tools/time_utils.h"
    
    #include "lenet.h"
    #include "mnist_data_proc.h"
    
    using namespace cacu;
    using namespace cacu_tools;
    
    void train_net()
    {
        int batch_size = 100;
    
        int max_iter = 5000;
    
    #if __USE_DEVICE__ == ON
    #if __PARALLELTYPE__ == __CUDA__
        cuda_set_device(0);
    
    #endif
    #endif
        //set random seed
        set_rand_seed();
    
        network *net = create_lenet(batch_size,train);
    
        sgd_solver *sgd = new sgd_solver(net);
        sgd->set_lr(0.01f);
        sgd->set_momentum(0.9f);
        sgd->set_weight_decay(0.0005f);
    
        string datapath = "/home/luhaofang/git/caffe/data/mnist/";
    
        std::ofstream logger(datapath + "loss.txt", ios::binary);
        logger.precision(std::numeric_limits<cacu::float_t>::digits10);
    
        string meanfile = datapath + "mean.binproto";
    
        vector<vec_t> full_data;
        vector<vec_i> full_label;
    
        load_data_bymean_mnist(datapath, meanfile, full_data, full_label);
        //load_data(datapath, full_data, full_label);
    
        blob *input_data = (blob*)net->input_blobs()->at(0);
        bin_blob *input_label = (bin_blob*)net->input_blobs()->at(1);
    
        int step_index = 0;
        time_utils *timer = new time_utils();
        unsigned long diff;
        for (int i = 1 ; i < max_iter; ++i)
        {
            timer->start();
    
            for (int j = 0 ; j < batch_size ; ++j)
            {
                if (step_index == kMNISTDataCount)
                    step_index = 0;
                input_data->copy2data(full_data[step_index], j);
                input_label->copy2data(full_label[step_index],j);
                step_index += 1;
            }
            
            sgd->train_iter(i);
            //cacu_print(net->get_op<inner_product_op>(net->op_count() - 2, CACU_INNERPRODUCT)->out_data<blob>()->s_data(), 10);
    
            timer->end();
    
            if(i % 10 == 0){
    
                LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, sgd->lr(), timer->get_time_span() / 1000);
                net->get_op<softmax_with_loss_op>(net->op_count() - 1, CACU_SOFTMAX_LOSS)->echo();
    
                logger << net->get_op<softmax_with_loss_op>(net->op_count() - 1, CACU_SOFTMAX_LOSS)->loss() << endl;
                logger.flush();
            }
    
            if(i % 4000 == 0)
                sgd->set_lr_iter(0.1f);
    
        }
        LOG_INFO("optimization is done!");
        net->save_weights(datapath + "lenet.model");
    
        vector<vec_t>().swap(full_data);
        vector<vec_i>().swap(full_label);
        logger.close();
        delete net;
        delete sgd;
    
        delete timer;
    
    #if __USE_DEVICE__ == ON
    #if __PARALLELTYPE__ == __CUDA__
        cuda_release();
    #endif
    #endif
    }

### imageNet

Inference running time cost:

-cpu

|   | ave(ms)  |  max(ms) |  min(ms) | acc |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| res18net  | 99  |  123 | 95  |  66.71%  |
| res50net  | 192  | 204  | 187  |  72.15% |
| vgg16net  | 702  | 732  | 679  |  66.41% |
| mobilenet  | 110  | 127  | 106  |  67.85% |

-gpu

|   | ave(ms)  |  max(ms) |  min(ms) | acc |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| res18net  | 8  |  8 | 8  |  66.87%  |
| res50net  | 18  | 19  | 18  |  71.80% |
| vgg16net  | 19  | 20  | 19  |  65.98% |
| mobilenet  | 32  | 37  | 32  |  67.73% |

All the models are trained without data argumentation.

- pre-trained [res18net](https://pan.baidu.com/s/1c1BgMXi) 
- pre-trained [res50net](https://pan.baidu.com/s/1bptuG95) 
- pre-trained [vgg16net](https://pan.baidu.com/s/1eR8tGLO) 
- pre-trained [mobilenet](https://pan.baidu.com/s/1c1IAAE8) 

vgg16net feature map demonstration.

![image](https://github.com/luhaofang/CACUE/blob/master/example/imagenet/img/pic.JPEG)
![image](https://github.com/luhaofang/CACUE/blob/master/example/imagenet/img/test.jpg)


### MTCNN (just demo, need modified)

This implementation is referred to [MTCNN](https://github.com/CongWeilin/mtcnn-caffe)

![image](https://github.com/luhaofang/CACUE/blob/master/example/mtcnn/test.jpg)

### DCGAN

DCGAN on cifar10 demonstration.

- 5000 iterations: ![image](https://github.com/luhaofang/CACUE/blob/master/example/gan/img/test_5000.jpg)

- 6000 iterations: ![image](https://github.com/luhaofang/CACUE/blob/master/example/gan/img/test_6000.jpg)

- 7000 iterations: ![image](https://github.com/luhaofang/CACUE/blob/master/example/gan/img/test_7000.jpg)

- 8000 iterations: ![image](https://github.com/luhaofang/CACUE/blob/master/example/gan/img/test_8000.jpg)

### CycleGAN

CycleGAN on imagenet dataset demonstration.

Loss function: sigmoid with cross entropy.
![image](https://github.com/luhaofang/CACUE/blob/master/example/cycle_gan/img/cyclegan_loss.png)

|  zebra->horse  |  horse->zebra |
| ------------ | ------------ |
| ![image](https://github.com/luhaofang/CACUE/blob/master/example/cycle_gan/img/n02391049_400.png)->![image](https://github.com/luhaofang/CACUE/blob/master/example/cycle_gan/img/n02381460_1260.png) |  ![image](https://github.com/luhaofang/CACUE/blob/master/example/cycle_gan/img/n02381460_1300.png)->![image](https://github.com/luhaofang/CACUE/blob/master/example/cycle_gan/img/n02391049_980.png) |
| ![image](https://github.com/luhaofang/CACUE/blob/master/example/cycle_gan/img/n02391049_5100.png)->![image](https://github.com/luhaofang/CACUE/blob/master/example/cycle_gan/img/n02381460_1350.png) |  ![image](https://github.com/luhaofang/CACUE/blob/master/example/cycle_gan/img/n02391049_260.png)->![image](https://github.com/luhaofang/CACUE/blob/master/example/cycle_gan/img/n02381460_8980.png) |
| ![image](https://github.com/luhaofang/CACUE/blob/master/example/cycle_gan/img/n02391049_9960.png)->![image](https://github.com/luhaofang/CACUE/blob/master/example/cycle_gan/img/n02381460_6920.png)  | 	![image](https://github.com/luhaofang/CACUE/blob/master/example/cycle_gan/img/n02381460_4530.png)->![image](https://github.com/luhaofang/CACUE/blob/master/example/cycle_gan/img/n02391049_180.png) |	

## References
[1] A Krizhevsky, I Sutskever, GE Hinton. [Imagenet classification with deep convolutional neural networks.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). 
    Advances in neural information processing systems, 2012: 1097-1105.
    
[2] Rastegari M, Ordonez V, Redmon J, et al. [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/pdf/1603.05279.pdf).
	arXiv preprint arXiv:1603.05279, 2016.

[3] S Ioffe, C Szegedy. [Batch normalization: Accelerating deep network training by reducing internal covariate shift.](https://arxiv.org/pdf/1502.03167v3.pdf).
    arXiv preprint arXiv:1502.03167, 2015.
	
[4] Courbariaux M, Bengio Y. [Binarynet: Training deep neural networks with weights and activations constrained to+ 1 or-1](https://arxiv.org/pdf/1602.02830.pdf). 
	arXiv preprint arXiv:1602.02830, 2016.
	
[5] Radford, Alec, Luke Metz, and Soumith Chintala. [Unsupervised representation learning with deep convolutional generative adversarial networks.](https://arxiv.org/pdf/1511.06434.pdf).
    arXiv preprint arXiv:1511.06434, 2015.

[6] Zhang K, Zhang Z, Li Z, Qiao Y. [Joint face detection and alignment using multitask cascaded convolutional networks.](https://arxiv.org/pdf/1604.02878.pdf).
    IEEE Signal Processing Letters, 2016 Oct;23(10):1499-503. 
    
[7] Howard, Andrew G., et al. [Mobilenets: Efficient convolutional neural networks for mobile vision applications.](https://arxiv.org/pdf/1704.04861.pdf).
    arXiv preprint arXiv:1704.04861, 2017.
    
[8] He, Kaiming, et al. [Deep residual learning for image recognition.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf).
    CVPR, 2016.
    
[9] Zhu, Jun-Yan, et al. [Unpaired image-to-image translation using cycle-consistent adversarial networks.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf).
    ICCV, 2017.

[10] Krizhevsky, Alex, Vinod Nair, and Geoffrey Hinton. [The CIFAR-10 dataset.]. online: http://www.cs.toronto.edu/kriz/cifar.html, 2014.

[11] LeCun, Yann, Corinna Cortes, and C. J. Burges. "MNIST handwritten digit database." AT&T Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, 2010.

[12] Deng, Jia, et al. [Imagenet: A large-scale hierarchical image database.]. CVPR, 2009.