#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include<boost/shared_ptr.hpp>

 namespace caffe {
class Blob {
public:
        Blob() : num_(0), channels_(0), height_(0), width_(0), count_(0),
                    data_(), diff_()  {}
        explicit Blob(const int num, const int channels,
                      const int height, const int width);
        virtual ~Blob();
        const float* data() const;
        const float* diff() const;

protected:
        boost::shared_ptr<float> data_;
        boost::shared_ptr<float> diff_;
        int num_;
        int channels_;
        int height_;
        int width_;
        int count_;
};
 }
#endif // CAFFE_BLOB_HPP_
