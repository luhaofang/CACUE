################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../cacu/math/cuda/activation_functions.cu \
../cacu/math/cuda/batch_functions.cu \
../cacu/math/cuda/loss_functions.cu \
../cacu/math/cuda/math_functions.cu \
../cacu/math/cuda/norm_functions.cu \
../cacu/math/cuda/pooling_functions.cu 

OBJS += \
./cacu/math/cuda/activation_functions.o \
./cacu/math/cuda/batch_functions.o \
./cacu/math/cuda/loss_functions.o \
./cacu/math/cuda/math_functions.o \
./cacu/math/cuda/norm_functions.o \
./cacu/math/cuda/pooling_functions.o 

CU_DEPS += \
./cacu/math/cuda/activation_functions.d \
./cacu/math/cuda/batch_functions.d \
./cacu/math/cuda/loss_functions.d \
./cacu/math/cuda/math_functions.d \
./cacu/math/cuda/norm_functions.d \
./cacu/math/cuda/pooling_functions.d 


# Each subdirectory must supply rules for building sources it contributes
cacu/math/cuda/%.o: ../cacu/math/cuda/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "cacu/math/cuda" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


