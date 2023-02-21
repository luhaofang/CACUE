################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../cacu/user_ops/combination_op.cc \
../cacu/user_ops/conditional_bn_op.cc \
../cacu/user_ops/conditional_in_op.cc \
../cacu/user_ops/conditional_projection_op.cc \
../cacu/user_ops/conv_transpose_op.cc \
../cacu/user_ops/deconvolution_op.cc \
../cacu/user_ops/injector_op.cc \
../cacu/user_ops/instance_normalize_op.cc \
../cacu/user_ops/ip_projection_op.cc \
../cacu/user_ops/normalization_op.cc \
../cacu/user_ops/triplet_loss_op.cc 

CC_DEPS += \
./cacu/user_ops/combination_op.d \
./cacu/user_ops/conditional_bn_op.d \
./cacu/user_ops/conditional_in_op.d \
./cacu/user_ops/conditional_projection_op.d \
./cacu/user_ops/conv_transpose_op.d \
./cacu/user_ops/deconvolution_op.d \
./cacu/user_ops/injector_op.d \
./cacu/user_ops/instance_normalize_op.d \
./cacu/user_ops/ip_projection_op.d \
./cacu/user_ops/normalization_op.d \
./cacu/user_ops/triplet_loss_op.d 

OBJS += \
./cacu/user_ops/combination_op.o \
./cacu/user_ops/conditional_bn_op.o \
./cacu/user_ops/conditional_in_op.o \
./cacu/user_ops/conditional_projection_op.o \
./cacu/user_ops/conv_transpose_op.o \
./cacu/user_ops/deconvolution_op.o \
./cacu/user_ops/injector_op.o \
./cacu/user_ops/instance_normalize_op.o \
./cacu/user_ops/ip_projection_op.o \
./cacu/user_ops/normalization_op.o \
./cacu/user_ops/triplet_loss_op.o 


# Each subdirectory must supply rules for building sources it contributes
cacu/user_ops/%.o: ../cacu/user_ops/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "cacu/solvers" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



