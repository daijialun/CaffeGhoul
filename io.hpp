#ifndef CAFFE_IO_HPP_
#define CAFFE_IO_HPP_

#include <google/protobuf/message.h>
#include <string>
#include "blob.hpp"
#include "caffe.pb.h"

namespace caffe {

void ReadProtoFromTextFile(const char* filename,
            google::protobuf::Message* proto);

}

#endif // CAFFE_IO_HPP_
