/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "FFT.hpp"
#include "FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Eigen>
#include <cmath>

namespace fluid {
namespace algorithm {

class TruePeak
{

  using ArrayXcd = Eigen::ArrayXcd;

public:
  TruePeak(index maxSize) : mFFT(maxSize), mIFFT(maxSize * 4) {}

  void init(index size, double sampleRate)
  {
    using namespace std;
    mSampleRate = sampleRate;
    mFFTSize = static_cast<index>(pow(2, ceil(log(size) / log(2))));
    mFactor = sampleRate < 96000 ? 4 : 2;
    mFFT.resize(mFFTSize);
    mIFFT.resize(mFFTSize * mFactor);
    mBuffer = ArrayXcd::Zero((mFFTSize * mFactor / 2) + 1);
  }

  double processFrame(const RealVectorView& input)
  {
    using namespace Eigen;
    ArrayXd in = _impl::asEigen<Array>(input);
    if (mSampleRate >= 192000) { return in.abs().maxCoeff(); }
    else
    {
      double   peak;
      ArrayXcd transform = mFFT.process(in);
      mBuffer.setZero();
      mBuffer.segment(0, transform.size()) = transform;
      ArrayXd result = mIFFT.process(mBuffer);
      ArrayXd scaled = result / mFFTSize;
      peak = scaled.abs().maxCoeff();
      return peak;
    }
  }

private:
  FFT      mFFT;
  IFFT     mIFFT;
  ArrayXcd mBuffer;
  double   mSampleRate{44100.0};
  index    mFactor{4};
  index    mFFTSize{1024};
};
} // namespace algorithm
} // namespace fluid
