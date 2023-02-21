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


#include "jpeg_encoder.h"

#include "imageio_utils.h"

using namespace std;
using namespace cacu;
using namespace cacu_tools;


void dct_trans(chars_t file_list, chars_t in_path, chars_t out_path)
{

	JpegEncoder *encoder = new JpegEncoder();

	ifstream is = ifstream(file_list);
	is.precision(numeric_limits<float>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", file_list.c_str());
	string file_ = "";
	vector<string> vec;
	string cls = "";
	string filename = "";
	blob *img_data = new blob(1, 3, 32, 32, 0, test);
	vector<chars_t> filenames;
	vector<chars_t> clss;
	while (getline(is, file_)) {
		LOG_INFO("%s", file_.c_str());
		vec = split(file_, " ");
		filename = vec[0];
		cls = vec[1];
		vec = split(filename, "/");
		filename = vec[1];
		filenames.push_back(filename);
		clss.push_back(cls);
	}
	is.close();

	for (int i = 0 ; i < filenames.size(); ++i){

		LOG_DEBUG("%s/%s", clss[i].c_str(), filenames[i].c_str());

		if(!encoder->readFromBMP((in_path + "/" + clss[i] + "/" + filenames[i]).c_str()))
			LOG_FATAL("file %s cannot be opened!", (in_path + "/" + clss[i] + "/" + filenames[i]).c_str());

		encoder->encodeToDCT(img_data->s_data(), 50);

		imageio_utils::imwrite(img_data, (out_path + "val/" + clss[i] + "/" + filenames[i]).c_str());

		//		cacu_print(img_data->s_data(), img_data->count());
	}

	delete encoder;
	delete img_data;
}

