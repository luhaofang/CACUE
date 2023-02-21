################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../tools/imageio_utils.cc \
../tools/serializer_utils.cc \
../tools/string_utils.cc \
../tools/time_utils.cc \
../tools/trainer.cc 

CC_DEPS += \
./tools/imageio_utils.d \
./tools/serializer_utils.d \
./tools/string_utils.d \
./tools/time_utils.d \
./tools/trainer.d 

OBJS += \
./tools/imageio_utils.o \
./tools/serializer_utils.o \
./tools/string_utils.o \
./tools/time_utils.o \
./tools/trainer.o 


# Each subdirectory must supply rules for building sources it contributes
tools/%.o: ../tools/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "tools" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


