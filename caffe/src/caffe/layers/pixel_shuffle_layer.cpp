#include <cfloat>
#include <vector>

#include "caffe/layers/pixel_shuffle_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PixelShuffleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  sigma_ = this->layer_param().pixel_shuffle_param().sigma();
  n_ = bottom[0]->num();
  c_ = bottom[0]->channels();
  h_ = bottom[0]->height();
  w_ = bottom[0]->width();
  CHECK(!(c_ % sigma_ % sigma_));
  shuffled_c = c_ / sigma_ / sigma_;
  shuffled_h = h_ * sigma_;
  shuffled_w = w_ * sigma_;
}

template <typename Dtype>
void PixelShuffleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape = bottom[0]->shape();
  shape[1] /= sigma_ * sigma_;
  shape[2] *= sigma_;
  shape[3] *= sigma_;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void PixelShuffleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // NOT IMPLEMENTED!!!
}

template <typename Dtype>
void PixelShuffleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // NOT IMPLEMENTED!!!
}

#ifdef CPU_ONLY
STUB_GPU(PixelShuffleLayer);
#endif

INSTANTIATE_CLASS(PixelShuffleLayer);
REGISTER_LAYER_CLASS(PixelShuffle);

}  // namespace caffe
