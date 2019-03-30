#pragma once

#include "../../algorithms/public/SpectralShape.hpp"
#include "../../data/TensorTypes.hpp"
#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../rt/BufferedProcess.hpp"
#include <tuple>

namespace fluid {
namespace client {

using algorithm::SpectralShape;

enum SpectralShapeParamIndex { kFFT, kMaxFFTSize };

auto constexpr SpectralShapeParams = defineParameters(
    FFTParam<kMaxFFTSize>("fft", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4), PowerOfTwo{})
  );

template <typename T>
class SpectralShapeClient
    : public FluidBaseClient<decltype(SpectralShapeParams),
                             SpectralShapeParams>,
      public AudioIn,
      public ControlOut {
  using HostVector = HostVector<T>;

public:
  SpectralShapeClient(ParamSetViewType &p)
      : FluidBaseClient(p), mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0) {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::controlChannelsOut(7);
    mDescriptors = FluidTensor<double, 1>(7);
  }

  void process(std::vector<HostVector> &input,
               std::vector<HostVector> &output) {
    using std::size_t;

    if (!input[0].data() || !output[0].data())
      return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= FluidBaseClient::controlChannelsOut() &&
           "Too few output channels");

    if (mWinSizeTracker.changed(get<kFFT>().frameSize())) {
      mMagnitude.resize(get<kFFT>().frameSize());
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          mAlgorithm.processFrame(mMagnitude, mDescriptors);
        });
    for (int i = 0; i < 7; ++i)
      output[i](0) = mDescriptors(i);
  }

  size_t latency() { return get<kFFT>().winSize(); }

private:
  ParameterTrackChanges<size_t> mWinSizeTracker;
  STFTBufferedProcess<ParamSetViewType, T, kFFT> mSTFTBufferedProcess;
  SpectralShape mAlgorithm{get<kMaxFFTSize>()};
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mDescriptors;
};

} // namespace client
} // namespace fluid
