################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../cacu/framework/bin_blob.cc \
../cacu/framework/blob.cc \
../cacu/framework/blob_base.cc \
../cacu/framework/cacu_op.cc \
../cacu/framework/layer.cc \
../cacu/framework/layer_block.cc \
../cacu/framework/model.cc \
../cacu/framework/network.cc \
../cacu/framework/weight.cc 

CC_DEPS += \
./cacu/framework/bin_blob.d \
./cacu/framework/blob.d \
./cacu/framework/blob_base.d \
./cacu/framework/cacu_op.d \
./cacu/framework/layer.d \
./cacu/framework/layer_block.d \
./cacu/framework/model.d \
./cacu/framework/network.d \
./cacu/framework/weight.d 

OBJS += \
./cacu/framework/bin_blob.o \
./cacu/framework/blob.o \
./cacu/framework/blob_base.o \
./cacu/framework/cacu_op.o \
./cacu/framework/layer.o \
./cacu/framework/layer_block.o \
./cacu/framework/model.o \
./cacu/framework/network.o \
./cacu/framework/weight.o 


# Each subdirectory must supply rules for building sources it contributes
cacu/framework/%.o: ../cacu/framework/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "cacu/framework" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


