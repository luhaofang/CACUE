################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../cacu/math/cpu/activation_functions_cpu.cc \
../cacu/math/cpu/loss_functions_cpu.cc \
../cacu/math/cpu/pooling_functions_cpu.cc 

CC_DEPS += \
./cacu/math/cpu/activation_functions_cpu.d \
./cacu/math/cpu/loss_functions_cpu.d \
./cacu/math/cpu/pooling_functions_cpu.d 

OBJS += \
./cacu/math/cpu/activation_functions_cpu.o \
./cacu/math/cpu/loss_functions_cpu.o \
./cacu/math/cpu/pooling_functions_cpu.o 


# Each subdirectory must supply rules for building sources it contributes
cacu/math/cpu/%.o: ../cacu/math/cpu/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "cacu/math/cpu" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


