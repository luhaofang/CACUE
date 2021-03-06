/*
Copyright (c) 2016, David lu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef OPERATORS_H_
#define OPERATORS_H_

#include "../config.h"

#include "../utils/check_utils.h"
#include "../utils/log.h"

#include "../math/math.h"

#include "../tensor/utils/device_data_utils.h"
#include "../tensor/utils/cpu_data_utils.h"
#include "../utils/data_printer.h"

#include "operator_base.h"
#include "loss_base_op.h"
#include "activate_base_op.h"
#include "conv_base_op.h"
#include "pooling_base_op.h"
#include "math_op.h"

#include "inner_product_op.hpp"
#include "convolution_op.hpp"

#include "batch_normalize_op.hpp"
#include "dropout_op.hpp"

#include "sum_elemwise_op.hpp"
#include "feature_combine_op.hpp"
#include "split_op.hpp"

#include "average_pooling_op.hpp"
#include "max_pooling_op.hpp"

#include "relu_op.hpp"
#include "prelu_op.hpp"
#include "leaky_relu_op.hpp"
#include "softmax_op.hpp"
#include "tanh_op.hpp"
#include "half_tanh_op.hpp"
#include "sigmoid_op.hpp"
#include "elu_op.hpp"
#include "selu_op.hpp"

#include "softmax_with_loss_op.hpp"
#include "mse_loss_op.hpp"
#include "abse_loss_op.hpp"
#include "sigmoid_with_loss_op.hpp"

#include "add_op.hpp"
#include "axpby_op.hpp"
#include "sub_op.hpp"
#include "scalex_op.hpp"

//#include "hinge_loss_op.h"

/***********************************/
/*        custom operators	       */
/***********************************/


#include "../user_ops/lstm_base_op.h"

#include "../user_ops/p_inner_product_op.hpp"
#include "../user_ops/deconvolution_op.hpp"
#include "../user_ops/conv_transpose_op.hpp"
#include "../user_ops/normalization_op.hpp"
#include "../user_ops/dilated_convolution_op.hpp"
#include "../user_ops/combination_op.hpp"
#include "../user_ops/triplet_loss_op.hpp"
#include "../user_ops/shuffle_channel_op.hpp"

#include "../user_ops/roi_pooling_op.hpp"
#include "../user_ops/injector_op.hpp"
#include "../user_ops/instance_normalize_op.hpp"
#include "../user_ops/primary_vector_op.hpp"



#endif
