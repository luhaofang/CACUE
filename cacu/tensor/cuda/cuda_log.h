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

#ifndef CUDA_LOG_H_
#define CUDA_LOG_H_

#include <stdio.h>

#include "../../config.h"

#if defined(__USE_DEVICE__) && defined(__PARALLELTYPE__)
#if  __USE_DEVICE__ == ON && __PARALLELTYPE__ == __CUDA__

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace cacu {

#define BLOCKNUM 1024
#define THREADNUM 256

//cuda log utilities
#define CUDA_LOG(level, res)   \
			{do{ fprintf(stderr,"[%s][%s %s:%d] %s\n",level, __TIME__, __FILE__, __LINE__,res);}while(0); exit(-1);};

#define CUDA_CHECK(res) \
		if(res!=cudaSuccess) {\
			switch(res) { \
				case cudaErrorMissingConfiguration: \
					CUDA_LOG("CUDA","The device function being invoked (usually via cudaLaunchKernel()) was not previously configured via the cudaConfigureCall() function."); \
					break; \
				case cudaErrorMemoryAllocation: \
					CUDA_LOG("CUDA","The API call failed because it was unable to allocate enough memory to perform the requested operation."); break; \
				case cudaErrorInitializationError: \
					CUDA_LOG("CUDA","The API call failed because the CUDA driver and runtime could not be initialized."); break; \
				case cudaErrorLaunchFailure: \
					CUDA_LOG("CUDA","An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. The device cannot be used until cudaThreadExit() is called. All existing device memory allocations are invalid and must be reconstructed if the program is to continue using CUDA."); break; \
				case cudaErrorPriorLaunchFailure: \
					CUDA_LOG("CUDA","Deprecated! This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release. This indicated that a previous kernel launch failed. This was previously used for device emulation of kernel launches."); break; \
				case cudaErrorLaunchTimeout: \
					CUDA_LOG("CUDA","This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device property kernelExecTimeoutEnabled for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."); break; \
				case cudaErrorLaunchOutOfResources: \
					CUDA_LOG("CUDA","This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to cudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count."); break; \
				case cudaErrorInvalidDeviceFunction: \
					CUDA_LOG("CUDA","The requested device function does not exist or is not compiled for the proper device architecture."); break; \
				case cudaErrorInvalidConfiguration: \
					CUDA_LOG("CUDA","This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See cudaDeviceProp for more device limitations."); break; \
				case cudaErrorInvalidDevice: \
					CUDA_LOG("CUDA","This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device."); break; \
				case cudaErrorInvalidValue: \
					CUDA_LOG("CUDA","This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values."); break; \
				case cudaErrorInvalidPitchValue: \
					CUDA_LOG("CUDA","This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch."); break; \
				case cudaErrorInvalidSymbol: \
					CUDA_LOG("CUDA","This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier."); break; \
				case cudaErrorMapBufferObjectFailed: \
					CUDA_LOG("CUDA","This indicates that the buffer object could not be mapped."); break; \
				case cudaErrorUnmapBufferObjectFailed: \
					CUDA_LOG("CUDA","This indicates that the buffer object could not be unmapped."); break; \
				case cudaErrorInvalidHostPointer: \
					CUDA_LOG("CUDA","This indicates that at least one host pointer passed to the API call is not a valid host pointer."); break; \
				case cudaErrorInvalidDevicePointer: \
					CUDA_LOG("CUDA","This indicates that at least one device pointer passed to the API call is not a valid device pointer."); break; \
				case cudaErrorInvalidTexture: \
					CUDA_LOG("CUDA","This indicates that the texture passed to the API call is not a valid texture."); break; \
				case cudaErrorInvalidTextureBinding: \
					CUDA_LOG("CUDA","This indicates that the texture binding is not valid. This occurs if you call cudaGetTextureAlignmentOffset() with an unbound texture."); break; \
				case cudaErrorInvalidChannelDescriptor: \
					CUDA_LOG("CUDA","This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of the formats specified by cudaChannelFormatKind, or if one of the dimensions is invalid."); break; \
				case cudaErrorInvalidMemcpyDirection: \
					CUDA_LOG("CUDA","This indicates that the direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind."); break; \
				case cudaErrorAddressOfConstant: \
					CUDA_LOG("CUDA","Deprecated! This error return is deprecated as of CUDA 3.1. Variables in constant memory may now have their address taken by the runtime via cudaGetSymbolAddress(). This indicated that the user has taken the address of a constant variable, which was forbidden up until the CUDA 3.1 release."); break; \
				case cudaErrorTextureFetchFailed: \
					CUDA_LOG("CUDA","Deprecated! This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release. This indicated that a texture fetch was not able to be performed. This was previously used for device emulation of texture operations."); break; \
				case cudaErrorTextureNotBound: \
					CUDA_LOG("CUDA","Deprecated! This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release. This indicated that a texture was not bound for access. This was previously used for device emulation of texture operations."); break; \
				case cudaErrorSynchronizationError: \
					CUDA_LOG("CUDA","Deprecated! This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release. This indicated that a synchronization operation had failed. This was previously used for some device emulation functions."); break; \
				case cudaErrorInvalidFilterSetting: \
					CUDA_LOG("CUDA","This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA."); break; \
				case cudaErrorInvalidNormSetting: \
					CUDA_LOG("CUDA","This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA."); break; \
				case cudaErrorMixedDeviceExecution: \
					CUDA_LOG("CUDA","Deprecated! This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release. Mixing of device and device emulation code was not allowed."); break; \
				case cudaErrorCudartUnloading: \
					CUDA_LOG("CUDA","This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down, at a point in time after CUDA driver has been unloaded."); break; \
				case cudaErrorUnknown: \
					CUDA_LOG("CUDA","This indicates that an unknown internal error has occurred."); break; \
				case cudaErrorNotYetImplemented: \
					CUDA_LOG("CUDA","Deprecated! This error return is deprecated as of CUDA 4.1. This indicates that the API call is not yet implemented. Production releases of CUDA will never return this error."); break; \
				case cudaErrorMemoryValueTooLarge: \
					CUDA_LOG("CUDA","Deprecated! This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release. This indicated that an emulated device pointer exceeded the 32-bit address range."); break; \
				case cudaErrorInvalidResourceHandle: \
					CUDA_LOG("CUDA","This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like cudaStream_t and cudaEvent_t."); break; \
				case cudaErrorNotReady: \
					CUDA_LOG("CUDA","This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than cudaSuccess (which indicates completion). Calls that may return this value include cudaEventQuery() and cudaStreamQuery()."); break; \
				case cudaErrorInsufficientDriver: \
					CUDA_LOG("CUDA","This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration. Users should install an updated NVIDIA display driver to allow the application to run."); break; \
				case cudaErrorSetOnActiveProcess: \
					CUDA_LOG("CUDA","This indicates that the user has called cudaSetValidDevices(), cudaSetDeviceFlags(), cudaD3D9SetDirect3DDevice(), cudaD3D10SetDirect3DDevice, cudaD3D11SetDirect3DDevice(), or cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by calling non-device management operations (allocating memory and launching kernels are examples of non-device management operations). This error can also be returned if using runtime/driver interoperability and there is an existing CUcontext active on the host thread."); break; \
				case cudaErrorInvalidSurface: \
					CUDA_LOG("CUDA","This indicates that the surface passed to the API call is not a valid surface."); break; \
				case cudaErrorNoDevice: \
					CUDA_LOG("CUDA","This indicates that no CUDA-capable devices were detected by the installed CUDA driver."); break; \
				case cudaErrorECCUncorrectable: \
					CUDA_LOG("CUDA","This indicates that an uncorrectable ECC error was detected during execution."); break; \
				case cudaErrorSharedObjectSymbolNotFound: \
					CUDA_LOG("CUDA","This indicates that a link to a shared object failed to resolve."); break; \
				case cudaErrorSharedObjectInitFailed: \
					CUDA_LOG("CUDA","This indicates that initialization of a shared object failed."); break; \
				case cudaErrorUnsupportedLimit: \
					CUDA_LOG("CUDA","This indicates that the cudaLimit passed to the API call is not supported by the active device."); break; \
				case cudaErrorDuplicateVariableName: \
					CUDA_LOG("CUDA","This indicates that multiple global or constant variables (across separate CUDA source files in the application) share the same string name."); break; \
				case cudaErrorDuplicateTextureName: \
					CUDA_LOG("CUDA","This indicates that multiple textures (across separate CUDA source files in the application) share the same string name."); break; \
				case cudaErrorDuplicateSurfaceName: \
					CUDA_LOG("CUDA","This indicates that multiple surfaces (across separate CUDA source files in the application) share the same string name."); break; \
				case cudaErrorDevicesUnavailable: \
					CUDA_LOG("CUDA","This indicates that all CUDA devices are busy or unavailable at the current time. Devices are often busy/unavailable due to use of cudaComputeModeExclusive, cudaComputeModeProhibited or when long running CUDA kernels have filled up the GPU and are blocking new work from starting. They can also be unavailable due to memory constraints on a device that already has active CUDA work being performed."); break; \
				case cudaErrorInvalidKernelImage: \
					CUDA_LOG("CUDA","This indicates that the device kernel image is invalid."); break; \
				case cudaErrorNoKernelImageForDevice: \
					CUDA_LOG("CUDA","This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration."); break; \
				case cudaErrorIncompatibleDriverContext: \
					CUDA_LOG("CUDA","This indicates that the current context is not compatible with this the CUDA Runtime. This can only occur if you are using CUDA Runtime/Driver interoperability and have created an existing Driver context using the driver API. The Driver context may be incompatible either because the Driver context was created using an older version of the API, because the Runtime API call expects a primary driver context and the Driver context is not primary, or because the Driver context has been destroyed. Please see Interactions with the CUDA Driver API for more information."); break; \
				case cudaErrorPeerAccessAlreadyEnabled: \
					CUDA_LOG("CUDA","This error indicates that a call to cudaDeviceEnablePeerAccess() is trying to re-enable peer addressing on from a context which has already had peer addressing enabled."); break; \
				case cudaErrorPeerAccessNotEnabled: \
					CUDA_LOG("CUDA","This error indicates that cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet via cudaDeviceEnablePeerAccess()."); break; \
				case cudaErrorDeviceAlreadyInUse: \
					CUDA_LOG("CUDA","This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread."); break; \
				case cudaErrorProfilerDisabled: \
					CUDA_LOG("CUDA","This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler."); break; \
				case cudaErrorProfilerNotInitialized: \
					CUDA_LOG("CUDA","Deprecated! This error return is deprecated as of CUDA 5.0. It is no longer an error to attempt to enable/disable the profiling via cudaProfilerStart or cudaProfilerStop without initialization."); break; \
				case cudaErrorProfilerAlreadyStarted: \
					CUDA_LOG("CUDA","Deprecated! This error return is deprecated as of CUDA 5.0. It is no longer an error to call cudaProfilerStart() when profiling is already enabled."); break; \
				case cudaErrorProfilerAlreadyStopped: \
					CUDA_LOG("CUDA","Deprecated! This error return is deprecated as of CUDA 5.0. It is no longer an error to call cudaProfilerStop() when profiling is already disabled."); break; \
				case cudaErrorAssert: \
					CUDA_LOG("CUDA","An assert triggered in device code during kernel execution. The device cannot be used again until cudaThreadExit() is called. All existing allocations are invalid and must be reconstructed if the program is to continue using CUDA."); break; \
				case cudaErrorTooManyPeers: \
					CUDA_LOG("CUDA","This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cudaEnablePeerAccess()."); break; \
				case cudaErrorHostMemoryAlreadyRegistered: \
					CUDA_LOG("CUDA","This error indicates that the memory range passed to cudaHostRegister() has already been registered."); break; \
				case cudaErrorHostMemoryNotRegistered: \
					CUDA_LOG("CUDA","This error indicates that the pointer passed to cudaHostUnregister() does not correspond to any currently registered memory region."); break; \
				case cudaErrorOperatingSystem: \
					CUDA_LOG("CUDA","This error indicates that an OS call failed."); break; \
				case cudaErrorPeerAccessUnsupported: \
					CUDA_LOG("CUDA","This error indicates that P2P access is not supported across the given devices."); break; \
				case cudaErrorLaunchMaxDepthExceeded: \
					CUDA_LOG("CUDA","This error indicates that a device runtime grid launch did not occur because the depth of the child grid would exceed the maximum supported number of nested grid launches."); break; \
				case cudaErrorLaunchFileScopedTex: \
					CUDA_LOG("CUDA","This error indicates that a grid launch did not occur because the kernel uses file-scoped textures which are unsupported by the device runtime. Kernels launched via the device runtime only support textures created with the Texture Object API's."); break; \
				case cudaErrorLaunchFileScopedSurf: \
					CUDA_LOG("CUDA","This error indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported by the device runtime. Kernels launched via the device runtime only support surfaces created with the Surface Object API's."); break; \
				case cudaErrorSyncDepthExceeded: \
					CUDA_LOG("CUDA","This error indicates that a call to cudaDeviceSynchronize made from the device runtime failed because the call was made at grid depth greater than than either the default (2 levels of grids) or user specified device limit cudaLimitDevRuntimeSyncDepth. To be able to synchronize on launched grids at a greater depth successfully, the maximum nested depth at which cudaDeviceSynchronize will be called must be specified with the cudaLimitDevRuntimeSyncDepth limit to the cudaDeviceSetLimit api before the host-side launch of a kernel using the device runtime. Keep in mind that additional levels of sync depth require the runtime to reserve large amounts of device memory that cannot be used for user allocations."); break; \
				case cudaErrorLaunchPendingCountExceeded: \
					CUDA_LOG("CUDA","This error indicates that a device runtime grid launch failed because the launch would exceed the limit cudaLimitDevRuntimePendingLaunchCount. For this launch to proceed successfully, cudaDeviceSetLimit must be called to set the cudaLimitDevRuntimePendingLaunchCount to be higher than the upper bound of outstanding launches that can be issued to the device runtime. Keep in mind that raising the limit of pending device runtime launches will require the runtime to reserve device memory that cannot be used for user allocations."); break; \
				case cudaErrorNotPermitted: \
					CUDA_LOG("CUDA","This error indicates the attempted operation is not permitted."); break; \
				case cudaErrorNotSupported: \
					CUDA_LOG("CUDA","This error indicates the attempted operation is not supported on the current system or device."); break; \
				case cudaErrorHardwareStackError: \
					CUDA_LOG("CUDA","Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."); break; \
				case cudaErrorIllegalInstruction: \
					CUDA_LOG("CUDA","The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."); break; \
				case cudaErrorMisalignedAddress: \
					CUDA_LOG("CUDA","The device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."); break; \
				case cudaErrorInvalidAddressSpace: \
					CUDA_LOG("CUDA","While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."); break; \
				case cudaErrorInvalidPc: \
					CUDA_LOG("CUDA","The device encountered an invalid program counter. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."); break; \
				case cudaErrorIllegalAddress: \
					CUDA_LOG("CUDA","The device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."); break; \
				case cudaErrorInvalidPtx: \
					CUDA_LOG("CUDA","A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device."); break; \
				case cudaErrorInvalidGraphicsContext: \
					CUDA_LOG("CUDA","This indicates an error with the OpenGL or DirectX context."); break; \
				case cudaErrorNvlinkUncorrectable: \
					CUDA_LOG("CUDA","This indicates that an uncorrectable NVLink error was detected during the execution."); break; \
				case cudaErrorStartupFailure: \
					CUDA_LOG("CUDA","This indicates an internal startup failure in the CUDA runtime."); break; \
				case cudaErrorApiFailureBase: \
					CUDA_LOG("CUDA","Deprecated! This error return is deprecated as of CUDA 4.1. Any unhandled CUDA driver error is added to this value and returned via the runtime. Production releases of CUDA should not return such errors."); break; \
				default : \
					break; \
			} \
		}

}

#endif
#endif


#endif
