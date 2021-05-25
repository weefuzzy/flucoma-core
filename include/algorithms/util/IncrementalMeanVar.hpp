#pragma once

#include <Eigen/Core>

namespace fluid {
namespace algorithm {
namespace _impl {

index incrementalMeanVariance(const Eigen::Ref<Eigen::ArrayXXd> data,
                              index                             lastSampleCount,
                              Eigen::Ref<Eigen::ArrayXd>        mean,
                              Eigen::Ref<Eigen::ArrayXd>        var)
{
  Eigen::ArrayXd rowSums = data.isNaN().select(0, data).colwise().sum();
  index          newSampleCount = data.rows();
  Eigen::ArrayXd lastSum = mean * lastSampleCount;
  if (mean.cols() > 0)
  {
    index updatedSampleCount = lastSampleCount + newSampleCount;
    mean = ((mean * lastSampleCount) + rowSums) / updatedSampleCount;

    if (var.rows() > 0)
    {
      assert(var.allFinite());
      Eigen::ArrayXd newUnnormalisedVar =
          ((data.isNaN().select(0, data).rowwise() - mean.transpose())
               .square()
               .colwise()
               .mean());

      double lastCountOverNewCount =
          static_cast<double>(lastSampleCount) / newSampleCount;
      var = ((var.square() * lastSampleCount) +
             (newUnnormalisedVar * newSampleCount) +
             lastCountOverNewCount / updatedSampleCount *
                 (lastSum / lastCountOverNewCount - rowSums).square()) /
            updatedSampleCount;

      var = var.sqrt();
    }
    return updatedSampleCount;
  }
  return lastSampleCount;
}

} // namespace _impl
} // namespace algorithm
} // namespace fluid
