#ifndef TEST_WEIGHT_HPP_
#define TEST_WEIGHT_HPP_


#include "../cacu/cacu.h"

#include "../cacu/framework/weight.h"
#include "../cacu/framework/gaussian_initializer.h"


#include "../cacu/math/norm_functions.h"
#include "../example/cifar10/cifar_quick_net.h"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

TEST_CASE("weight") {
	SECTION("weight functions test"){

	set_rand_seed();

	weight w("test", 1, 3, 3, 3, test);

	for(int i = 0 ; i < 10; ++i){
		w.set_init_type(gaussian, 1);
		cacu_print(w.s_data(),27);
	}

	//w.set_init_type(gaussian, 1);
	cacu_tanh(w.s_data(), w.count(), w.s_data());
	cacu_print(w.s_data(),27);

	w.set_init_type(constant, 0.1);
	cacu_print(w.s_data(),27);

	w.set_init_type(xavier, 0.1);
	cacu_print(w.s_data(),27);

	w.set_init_type(msra, 0.1);
	cacu_print(w.s_data(),27);


	weight *w_ = new weight("test", 1,3,3,3,test);
	gaussian_initializer *gi = new gaussian_initializer(0, 10);
	gi->init_blob((blob *&)w_);

	cacu_print(w_->s_data(),27);

}

}



TEST_CASE("l2_norm") {
	SECTION("weight functions test"){

	blob w(1, 32, 3, 3, 0, test);

	w.set_init_type(gaussian, 0.1);
	cacu_print(w.s_data(),w.count());

	float norm = 0;

	cacu_norm(w.s_data(), w.count(), 10e-12, norm_l2, &norm);
	cacu_print(w.s_data(),w.count());

}
}


#endif
