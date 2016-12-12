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

#pragma once

#include "../../utils/cacu_allocator.h"

namespace mycnn{

	class operator_base
	{

	public:

		operator_base(blob_base *&data, args *&args_){
			
			s_blob = data;
			_args = args_;		

		};

		operator_base(blobs *&data, args *&args_){

			s_blobs = data;
			_args = args_;
		};

		~operator_base(){

			delete _args;			
		};

		virtual const void op() = 0;

		virtual const void check() = 0;

		virtual const void grad(const solver_base *&solver_base) = 0;

		virtual const void load(std::ifstream &is) = 0;

		virtual const void save(std::ostream &os) = 0;

		virtual const void echo() = 0;

		inline blob_base *&out_data(){ return o_blob; }

		inline blobs *&in_datas(){ return s_blobs; };

		

	protected:

		blobs *s_blobs;

		blob_base *s_blob;

		blob_base *o_blob;

		args *_args;

	private:

		
	};
}
