/*
 * test_cacu_op.hpp
 *
 *  Created on: May 14, 2018
 *      Author: haofang
 */

#ifndef TEST_NETWORK_HPP_
#define TEST_NETWORK_HPP_

#include <time.h>

#include "../cacu/framework/network.h"
#include "../cacu/framework/gframework/cacu_graphic.h"
#include "../tools/prune_utils.h"


#include "../example/imagenet/resnet_18.h"
#include "../example/cifar10/cifar_quick_net.h"

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

TEST_CASE("network") {
	SECTION("network test"){
		network *net = create_cifar_quick_net(1,test);//create_res20net(1, test);
		net->serialize_model("/Users/seallhf/Documents/datasets/cifar10/cifar10_quick.ch");
		//net->load_weights("/Users/seallhf/Documents/datasets/cifar10/res18net_original.model");
		//net->save_weights("/Users/seallhf/Documents/datasets/cifar10/res18net.model");
	}
}

TEST_CASE("phrase_network") {
	SECTION("network test"){
		blobs *inputs = new blobs();
		inputs->push_back(new blob(1, 3, 224, 224, 0, test));
		network *net = phrase_network_for_pruning("/Users/seallhf/Documents/datasets/cifar10/res18net_cifar.ch", inputs);
		LOG_DEBUG("fuck!");
		net->set_is_use_bias(false);

		net->load_weights("/Users/seallhf/Documents/datasets/cifar10/res18net_40000_positive.model");
		net->network_pruning("/Users/seallhf/Documents/datasets/cifar10/res18net_cifar_pruned.ch");
		net->save_weights("/Users/seallhf/Documents/datasets/cifar10/res18net_40000_positive_pruned.model");
	}
}

TEST_CASE("prune_network") {
	SECTION("network test"){
		network *net = create_cifar_quick_net(1,test);
		prune_model(net,"/Users/seallhf/Documents/datasets/cifar10/cifar10_quick_test_0.96.model",
				"/Users/seallhf/Documents/datasets/cifar10/cifar10_quick_pruned.ch",
				"/Users/seallhf/Documents/datasets/cifar10/cifar10_quick_test_pruned.model");
	}
}

TEST_CASE("phrase_network_cifar") {
	SECTION("network test"){
		blobs *inputs = new blobs();
		inputs->push_back(new blob(1, 3, 32, 32, 0, test));
		network *net = phrase_network("/Users/seallhf/Documents/datasets/cifar10/cifar10_quick.ch", inputs);
		LOG_DEBUG("fuck!");
		net->set_is_use_bias(false);

		net->load_weights("/Users/seallhf/Documents/datasets/cifar10/cifar10_quick_test_0.96.model");
		net->network_pruning("/Users/seallhf/Documents/datasets/cifar10/cifar10_quick_pruned.ch");
		net->save_weights("/Users/seallhf/Documents/datasets/cifar10/cifar10_quick_test_pruned.model");
	}
}




#endif /* TEST_DECONV_HPP_ */
