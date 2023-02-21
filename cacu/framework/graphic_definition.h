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

#ifndef GRAPHIC_DEFINITION_H_
#define GRAPHIC_DEFINITION_H_

namespace cacu{

struct graphic_edge{

	//operator node index;
	int from_nodeId;
	//blob index in s_blob;
	int from_blobId;

	//operator node index;
	int to_nodeId;
	//blob index in s_blob;
	int to_blobId;

	graphic_edge(int from_nodeId_, int from_blobId_, int to_nodeId_, int to_blobId_){
		from_nodeId = from_nodeId_;
		from_blobId = from_blobId_;
		to_nodeId = to_nodeId_;
		to_blobId = to_blobId_;

	}

	void serialize(std::ostream &os){
		os.write((char*)(&from_nodeId), sizeof(int));
		os.write((char*)(&from_blobId), sizeof(int));
		os.write((char*)(&to_nodeId), sizeof(int));
		os.write((char*)(&to_blobId), sizeof(int));
	}

	void load(std::istream &is){
		is.read(reinterpret_cast<char*>(&from_nodeId), sizeof(int));
		is.read(reinterpret_cast<char*>(&from_blobId), sizeof(int));
		is.read(reinterpret_cast<char*>(&to_nodeId), sizeof(int));
		is.read(reinterpret_cast<char*>(&to_blobId), sizeof(int));
	}

};

}

#endif
