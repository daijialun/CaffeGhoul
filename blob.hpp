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
          void Reshape(const int num, const int height,
                const int width, const int channels);
        virtual ~Blob();
        const float* data() const;
        const float* diff() const;
        float* mutable_data();
        float* mutable_diff();
        inline int num() const { return num_; }
        inline int channels() const { return channels_; }
        inline int height() const { return height_; }
        inline int width() const { return width_; }
        inline int count() const {return count_; }
        inline int offset(const int n, const int c = 0, const int h = 0, const int w = 0) const {
            return ((n * channels_ + c) * height_ + h) * width_ + w;
        }

        // Copy from source. If copy_diff is false, we copy the data; if copy_diff is true, we copy the diff.
        void CopyFrom(const Blob& source, bool copy_diff = false, bool reshape = false);
        void Update();

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
