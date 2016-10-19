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
        return (const float*)data_.get();
}

const float* Blob::diff() const  {
        CHECK(diff_);
        return (const float*)diff_.get();
}

float* Blob::mutable_data()   {
        CHECK(data_);
        return data_.get();
}

float* Blob::mutable_diff() {
        CHECK(diff_);
        return  diff_.get();
}


void Blob::Reshape(const int num, const int channels, const int height, const int width) {
        CHECK_GE(num, 0);
        CHECK_GE(channels, 0);
        CHECK_GE(height, 0);
        CHECK_GE(width, 0);
        num_ = num;
        channels_ = channels;
        height_ = height;
        width_ = width;
        count_ = num_ * channels_ * height_ * width_;
        if (count_) {
            data_.reset( new float[count_] );
            diff_.reset( new float[count_] );
        }
}

void Blob::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
        if (num_ != source.num() || channels_ != source.channels() ||
            height_ != source.height() || width_ != source.width()) {
                if ( reshape ) {
                        Reshape(source.num(), source.channels(), source.height(), source.width());
                } else {
                        LOG(FATAL) << "Trying to copy blobs of different sizes.";
                }
        }
        /*if ( copy_diff ) {
                memcpy(diff_->mutable_data(), source.diff(), sizeof(float) * count_);
        } else {
                memcpy(data_->mutable_data(), source.data(), sizeof(float) * count_);
        }*/
}

}
