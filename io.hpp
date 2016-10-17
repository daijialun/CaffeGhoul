#ifndef CAFFE_IO_HPP_
#define CAFFE_IO_HPP_

#include <google/protobuf/message.h>
#include <string>
#include "blob.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
        void ReadProtoFromTextFile(const char* filename,
    ::google::protobuf::Message* proto) {
  int fd = open(filename, O_RDONLY);
  FileInputStream* input = new FileInputStream(fd);
  CHECK(google::protobuf::TextFormat::Parse(input, proto));
  delete input;
  close(fd);
}
}

#endif // CAFFE_IO_HPP_
