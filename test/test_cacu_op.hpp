/*
 * test_cacu_op.hpp
 *
 *  Created on: May 14, 2018
 *      Author: haofang
 */

#ifndef TEST_CACU_OP_HPP_
#define TEST_CACU_OP_HPP_


#include "../cacu/cacu.h"

#include "../cacu/framework/cacu_op.h"

#include <time.h>

#include "../tools/serializer_utils.h"

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

TEST_CASE("cacu_op") {
	SECTION("cacu_op dynamic test"){

//		blob *data = new blob(1, 3, 5, 5, 1,test);
//		cacu_print(data->s_data(),data->count());
//		cacu_op *conv = new cacu_op(CACU_CONVOLUTION, data, new data_args(10, 3, 1, 1, 3));
//		conv->_CREATE_OP();
//		conv->get_param(0)->set_init_type(gaussian,0.1);
//		conv->forward();
//		cacu_print(conv->get_param(0)->s_data(),conv->get_param(0)->count());
//		cacu_op *relu = new cacu_op(CACU_RELU);
//		*conv >> relu;
//		relu->_CREATE_OP();
//		relu->forward();
//		conv->get_oblob<blob>()->blob_size();
//		cacu_print(relu->get_oblob<blob>()->s_data(),relu->get_oblob<blob>()->count());

	}
}



#endif /* TEST_DECONV_HPP_ */
