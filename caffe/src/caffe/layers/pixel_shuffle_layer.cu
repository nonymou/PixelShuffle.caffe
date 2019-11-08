#include <cfloat>
#include <vector>

#include "caffe/layers/pixel_shuffle_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PixelShuffleForward(const int nthreads, const Dtype* bottom_data, Dtype* top_data, int sigma,
    int bottom_h, int bottom_w, int bottom_c, int shuffled_channels, int shuffled_height, int shuffled_width) {
  CUDA_KERNEL_LOOP(index, nthreads) {

    const int w = index % shuffled_width;
    const int h = (index / shuffled_width) % shuffled_height;
    const int c = (index / shuffled_width / shuffled_height) % shuffled_channels;
    const int n = index / shuffled_width / shuffled_height / shuffled_channels;
    int h0 = h / sigma;
    int w0 = w / sigma;
    int c0 = (h % sigma * sigma + w % sigma) * shuffled_channels + c;
    int offset = ((n * bottom_c + c0) * bottom_h + h0) * bottom_w + w0;

    top_data[index] = bottom_data[offset];
  }
}

template <typename Dtype>
void PixelShuffleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  PixelShuffleForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, top_data, sigma_,
      h_, w_, c_, shuffled_c, shuffled_h, shuffled_w);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void PixelShuffleBackward(const int nthreads, Dtype* bottom_diff, const Dtype* top_diff, int sigma,
                                    int bottom_h, int bottom_w, int bottom_c, int shuffled_channels, int shuffled_height, int shuffled_width) {
  CUDA_KERNEL_LOOP(index, nthreads) {

    const int w = index % shuffled_width;
    const int h = (index / shuffled_width) % shuffled_height;
    const int c = (index / shuffled_width / shuffled_height) % shuffled_channels;
    const int n = index / shuffled_width / shuffled_height / shuffled_channels;
    int h0 = h / sigma;
    int w0 = w / sigma;
    int c0 = (h % sigma * sigma + w % sigma) * shuffled_channels + c;
    int offset = ((n * bottom_c + c0) * bottom_h + h0) * bottom_w + w0;

    bottom_diff[offset] = top_diff[index];
  }
}

template <typename Dtype>
void PixelShuffleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    const int count = top[0]->count();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    PixelShuffleBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_diff, top_diff, sigma_,
        h_, w_, c_, shuffled_c, shuffled_h, shuffled_w);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PixelShuffleLayer);

}  // namespace caffe
