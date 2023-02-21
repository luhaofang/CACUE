################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../example/cifar10/data_proc.cc \
../example/cifar10/layer_creater.cc 

CC_DEPS += \
./example/cifar10/data_proc.d \
./example/cifar10/layer_creater.d 

OBJS += \
./example/cifar10/data_proc.o \
./example/cifar10/layer_creater.o 


# Each subdirectory must supply rules for building sources it contributes
example/cifar10/%.o: ../example/cifar10/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "example/cifar10" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


