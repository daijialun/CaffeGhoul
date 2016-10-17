#include <iostream>
#include <glog/logging.h>
//#include "proto/caffe.bp.h"
#include "blob.hpp"

namespace caffe {

Blob::Blob(const int num, const int channels, const int height, const int width)  {
        num_ = num;
        channels_ = channels;
        height_ = height;
        width_ = width;
        count_ = num_ * channels_ * height_ * width_;
        if( count_ )  {
                data_.reset( new float[count_] );
                diff_.reset( new float[count_] );
        }
}

const float* Blob::data() const  {
        CHECK(data_);
        return data_.get();
}

const float* Blob::diff() const  {
        CHECK(diff_);
        return diff_.get();
}
}
