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

#ifndef DATA_PRINTER_H_
#define DATA_PRINTER_H_


#include <stdio.h>
#include <iostream>
#include <ostream>

using namespace std;

namespace cacu{


inline void CACUE_PRINTER()
{
	cout << " ______     ______     ______     __  __     ______    " << endl;
	cout << "/\\  ___\\   /\\  __ \\   /\\  ___\\   /\\ \\/\\ \\   /\\  ___\\   " << endl;
	cout << "\\ \\ \\____  \\ \\  __ \\  \\ \\ \\____  \\ \\ \\_\\ \\  \\ \\  __\\   " << endl;
	cout << " \\ \\_____\\  \\ \\_\\ \\_\\  \\ \\_____\\  \\ \\_____\\  \\ \\_____\\ " << endl;
	cout << "  \\/_____/   \\/_/\\/_/   \\/_____/   \\/_____/   \\/_____/ " << endl;
	cout << "                                                       " << endl;
}

template<typename DTYPE>
inline void cacu_print(DTYPE *data, size_t length)
{
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	device_print(data,length);
#endif
#else
	cacu_print_cpu(data, length);
#endif
}


inline void cacu_bprint(blob *data_, phase_type phase_)
{
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	for(int n =0 ; n < data_->num();++n)
		for(int c = 0; c < data_->channel(); ++c)
			for(int h = 0; h < data_->height(); ++h){
				if(phase_ == train)
					device_print(data_->p_diff(n) + c * data_->channel_length() + h * data_->height(), data_->width());
				else
					device_print(data_->p_data(n) + c * data_->channel_length() + h * data_->height(), data_->width());
			}
#endif
#else
	for(int n =0 ; n < data_->num();++n)
		for(int c = 0; c < data_->channel(); ++c)
			for(int h = 0; h < data_->height(); ++h){
				if(phase_ == train)
					cacu_print_cpu(data_->p_diff(n) + c * data_->channel_length() + h * data_->height(), data_->width());
				else
					cacu_print_cpu(data_->p_data(n) + c * data_->channel_length() + h * data_->height(), data_->width());
			}
#endif
}

}


#endif
