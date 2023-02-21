################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../cacu/solvers/adam_solver.cc \
../cacu/solvers/rmsprop_solver.cc \
../cacu/solvers/sgd_solver.cc \
../cacu/solvers/solver_base.cc 

CC_DEPS += \
./cacu/solvers/adam_solver.d \
./cacu/solvers/rmsprop_solver.d \
./cacu/solvers/sgd_solver.d \
./cacu/solvers/solver_base.d 

OBJS += \
./cacu/solvers/adam_solver.o \
./cacu/solvers/rmsprop_solver.o \
./cacu/solvers/sgd_solver.o \
./cacu/solvers/solver_base.o 


# Each subdirectory must supply rules for building sources it contributes
cacu/solvers/%.o: ../cacu/solvers/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "cacu/solvers" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda/bin/nvcc -I/usr/include/openblas -I/usr/include -I/usr/local/cuda/include -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


