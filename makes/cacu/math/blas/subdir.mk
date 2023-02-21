################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../cacu/math/blas/cublas_utils.cc 

CC_DEPS += \
./cacu/math/blas/cublas_utils.d 

OBJS += \
./cacu/math/blas/cublas_utils.o 


# Each subdirectory must supply rules for building sources it contributes
cacu/math/blas/%.o: ../cacu/math/blas/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "cacu/math/blas" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


