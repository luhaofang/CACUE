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

namespace mycnn{

	class math_operator_base : public operator_base
	{

	public:

		math_operator_base(blobs *&data, math_args *&args_) : operator_base(data){

			o_blob = data->at(data->size()-1);
			_args = args_;

		};

		math_operator_base(blob_base *&data, math_args *&args_) : operator_base(data){

			o_blob = data;
			_args = args_;
		};

		~math_operator_base(){

		};

		virtual const void check() override{
			return;
		}

		inline virtual const void op() override {

		}


		inline virtual const void grad() override {

		}

		virtual const void load(std::ifstream& is) override {
			return;
		}

		virtual const void save(std::ostream& os) override {
			return;
		}

		virtual const void echo() override {
			return;
		}

		inline virtual const void LOOP_INIT_DATA_() override {

			//caution: here we DON'T reset every data blob, reset will be done by math_op caller
		}

	protected:

		math_args *_args;

	private:


	};
};
