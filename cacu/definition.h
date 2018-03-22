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

#ifndef DEFINITION_H_
#define DEFINITION_H_

#include <stdio.h>
#include <vector>
#include <map>
#include <string>

#include <iosfwd>
#include <fstream>
#include <istream>
#include <ostream>
#include <iostream>

namespace cacu{

#ifdef _WIN32

#define _MAX_INT_ INT_MAX
#define _MIN_FLT_ FLT_MIN

#elif defined(__linux)

#define _MAX_INT_ __INT_MAX__
#define _MIN_FLT_ __FLT_MIN__

#endif

using namespace std;

typedef float float_t;

typedef int dsize_t;

typedef string chars_t;

typedef map<chars_t, float_t> CACU_ARGS;

typedef vector<float_t> vec_t;

typedef vector<int> vec_i;

typedef vector<float_t*> vec_f;


#ifndef ON
#define ON 1
#endif

#ifndef OFF
#define OFF	0
#endif

}


#endif
