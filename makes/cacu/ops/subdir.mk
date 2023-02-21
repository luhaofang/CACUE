################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../cacu/ops/abse_loss_op.cc \
../cacu/ops/add_op.cc \
../cacu/ops/average_pooling_op.cc \
../cacu/ops/axpby_op.cc \
../cacu/ops/batch_normalize_op.cc \
../cacu/ops/convolution_op.cc \
../cacu/ops/dropout_op.cc \
../cacu/ops/elu_op.cc \
../cacu/ops/feature_combine_op.cc \
../cacu/ops/half_tanh_op.cc \
../cacu/ops/hinge_loss_op.cc \
../cacu/ops/inner_product_op.cc \
../cacu/ops/leaky_relu_op.cc \
../cacu/ops/max_pooling_op.cc \
../cacu/ops/mean_op.cc \
../cacu/ops/mse_loss_op.cc \
../cacu/ops/mul_op.cc \
../cacu/ops/prelu_op.cc \
../cacu/ops/relu_op.cc \
../cacu/ops/scalex_op.cc \
../cacu/ops/selu_op.cc \
../cacu/ops/sigmoid_op.cc \
../cacu/ops/sigmoid_with_loss_op.cc \
../cacu/ops/softmax_op.cc \
../cacu/ops/softmax_with_loss_op.cc \
../cacu/ops/softplus_op.cc \
../cacu/ops/split_op.cc \
../cacu/ops/sub_op.cc \
../cacu/ops/sum_elemwise_op.cc \
../cacu/ops/tanh_op.cc \
../cacu/ops/un_pooling_op.cc 

CC_DEPS += \
./cacu/ops/abse_loss_op.d \
./cacu/ops/add_op.d \
./cacu/ops/average_pooling_op.d \
./cacu/ops/axpby_op.d \
./cacu/ops/batch_normalize_op.d \
./cacu/ops/convolution_op.d \
./cacu/ops/dropout_op.d \
./cacu/ops/elu_op.d \
./cacu/ops/feature_combine_op.d \
./cacu/ops/half_tanh_op.d \
./cacu/ops/hinge_loss_op.d \
./cacu/ops/inner_product_op.d \
./cacu/ops/leaky_relu_op.d \
./cacu/ops/max_pooling_op.d \
./cacu/ops/mean_op.d \
./cacu/ops/mse_loss_op.d \
./cacu/ops/mul_op.d \
./cacu/ops/prelu_op.d \
./cacu/ops/relu_op.d \
./cacu/ops/scalex_op.d \
./cacu/ops/selu_op.d \
./cacu/ops/sigmoid_op.d \
./cacu/ops/sigmoid_with_loss_op.d \
./cacu/ops/softmax_op.d \
./cacu/ops/softmax_with_loss_op.d \
./cacu/ops/softplus_op.d \
./cacu/ops/split_op.d \
./cacu/ops/sub_op.d \
./cacu/ops/sum_elemwise_op.d \
./cacu/ops/tanh_op.d \
./cacu/ops/un_pooling_op.d 

OBJS += \
./cacu/ops/abse_loss_op.o \
./cacu/ops/add_op.o \
./cacu/ops/average_pooling_op.o \
./cacu/ops/axpby_op.o \
./cacu/ops/batch_normalize_op.o \
./cacu/ops/convolution_op.o \
./cacu/ops/dropout_op.o \
./cacu/ops/elu_op.o \
./cacu/ops/feature_combine_op.o \
./cacu/ops/half_tanh_op.o \
./cacu/ops/hinge_loss_op.o \
./cacu/ops/inner_product_op.o \
./cacu/ops/leaky_relu_op.o \
./cacu/ops/max_pooling_op.o \
./cacu/ops/mean_op.o \
./cacu/ops/mse_loss_op.o \
./cacu/ops/mul_op.o \
./cacu/ops/prelu_op.o \
./cacu/ops/relu_op.o \
./cacu/ops/scalex_op.o \
./cacu/ops/selu_op.o \
./cacu/ops/sigmoid_op.o \
./cacu/ops/sigmoid_with_loss_op.o \
./cacu/ops/softmax_op.o \
./cacu/ops/softmax_with_loss_op.o \
./cacu/ops/softplus_op.o \
./cacu/ops/split_op.o \
./cacu/ops/sub_op.o \
./cacu/ops/sum_elemwise_op.o \
./cacu/ops/tanh_op.o \
./cacu/ops/un_pooling_op.o 


# Each subdirectory must supply rules for building sources it contributes
cacu/ops/%.o: ../cacu/ops/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "cacu/solvers" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



