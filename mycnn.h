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

#pragma once

#include "core/utils/data_defination.h"
#include "core/utils/utils.h"
#include "core/utils/log.h"
#include "core/utils/configs.h"

#include "core/utils/args_base.h"
#include "core/utils/math_args_base.h"

#include "core/framework/args.h"
#include "core/framework/math_args.h"
#include "core/framework/blob_base.h"
#include "core/framework/blob.h"
#include "core/framework/em_blob.h"
#include "core/framework/bin_blob.h"
#include "core/framework/em_bin_blob.h"
#include "core/framework/blobs.h"
#include "core/framework/weight.h"

#include "core/utils/cacu_allocator.h"
#include "core/utils/blob_ops.h"

#include "core/math/matrix.h"
#include "core/math/cube.h"

#include "core/ops/operators.h"

#include "core/framework/layer_base.h"
#include "core/framework/layer.h"
#include "core/framework/em_layer.h"
#include "core/framework/layer_block.h"
#include "core/framework/network.h"
#include "core/framework/layer_creator.h"

#include "core/solvers/solvers.h"

