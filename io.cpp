#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <glog/logging.h>
#include "io.hpp"

namespace caffe {

void ReadProtoFromTextFile(const char* filename,
            google::protobuf::Message* proto)  {
        int fd = open(filename, O_RDONLY);
        google::protobuf::io::FileInputStream* input = new google::protobuf::io::FileInputStream(fd);
        CHECK(google::protobuf::TextFormat::Parse(input, proto));
        delete input;
        close(fd);
}

}
