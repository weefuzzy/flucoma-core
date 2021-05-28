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

#include "DataSetClient.hpp"
#include "NRTClient.hpp"
#include "../../algorithms/public/UMAP.hpp"

namespace fluid {
namespace client {
namespace umap {

enum {
  kNumDimensions,
  kNumNeighbors,
  kMinDistance,
  kNumIter,
  kLearningRate,
  kInputBuffer,
  kOutputBuffer
};

constexpr auto UMAPParams = defineParameters(
    LongParam("numDimensions", "Target Number of Dimensions", 2, Min(1)),
    LongParam("numNeighbours", "Number of Nearest Neighbours", 15, Min(1)),
    FloatParam("minDist", "Minimum Distance", 0.1, Min(0)),
    LongParam("iterations", "Number of Iterations", 200, Min(1)),
    FloatParam("learnRate", "Learning Rate", 0.1, Min(0.0), Max(1.0)),
    BufferParam("inputPointBuffer", "Input Point Buffer"),
    BufferParam("predictionBuffer", "Prediction Buffer"));

class UMAPClient : public FluidBaseClient,
                   AudioIn,
                   ControlOut,
                   ModelObject,
                   public DataClient<algorithm::UMAP>
{

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using StringVector = FluidTensor<string, 1>;

  using ParamDescType = decltype(UMAPParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto getParameterDescriptors() { return UMAPParams; }

  UMAPClient(ParamSetViewType& p) : mParams(p)
  {
    audioChannelsIn(1);
    controlChannelsOut(1);
  }

  MessageResult<void> fitTransform(DataSetClientRef sourceClient,
                                   DataSetClientRef destClient)
  {
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (!srcPtr || !destPtr) return Error(NoDataSet);
    auto src = srcPtr->getDataSet();
    auto dest = destPtr->getDataSet();
    if (src.size() == 0) return Error(EmptyDataSet);
    if (get<kNumNeighbors>() > src.size())
      return Error("Number of Neighbours is larger than dataset");
    FluidDataSet<string, double, 1> result;
    result = mAlgorithm.train(src, get<kNumNeighbors>(), get<kNumDimensions>(),
                              get<kMinDistance>(), get<kNumIter>(),
                              get<kLearningRate>());
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<void> fit(DataSetClientRef sourceClient)
  {
    auto srcPtr = sourceClient.get().lock();
    if (!srcPtr) return Error(NoDataSet);
    auto src = srcPtr->getDataSet();
    if (src.size() == 0) return Error(EmptyDataSet);
    if (get<kNumNeighbors>() > src.size())
      return Error("Number of Neighbours is larger than dataset");
    StringVector                    ids{src.getIds()};
    FluidDataSet<string, double, 1> result;
    result = mAlgorithm.train(src, get<kNumNeighbors>(), get<kNumDimensions>(),
                              get<kMinDistance>(), get<kNumIter>(),
                              get<kLearningRate>());
    return OK();
  }

  MessageResult<void> transform(DataSetClientRef sourceClient,
                                DataSetClientRef destClient)
  {
    auto srcPtr = sourceClient.get().lock();
    auto destPtr = destClient.get().lock();
    if (!srcPtr || !destPtr) return Error(NoDataSet);
    auto src = srcPtr->getDataSet();
    auto dest = destPtr->getDataSet();
    if (src.size() == 0) return Error(EmptyDataSet);
    if (!mAlgorithm.initialized()) return Error(NoDataFitted);
    if (get<kNumDimensions>() != mAlgorithm.dims())
      return Error("Wrong target number of dimensions");
    if (src.pointSize() != mAlgorithm.inputDims()) return Error(WrongPointSize);
    StringVector                    ids{src.getIds()};
    FluidDataSet<string, double, 1> result;
    result = mAlgorithm.transform(src, get<kNumIter>(), get<kLearningRate>());
    destPtr->setDataSet(result);
    return OK();
  }

  MessageResult<void> transformPoint(BufferPtr in, BufferPtr out)
  {
    index inSize = mAlgorithm.inputDims();
    index outSize = mAlgorithm.dims();
    if (!mAlgorithm.initialized()) return Error(NoDataFitted);
    if (get<kNumDimensions>() != outSize)
      return Error("Wrong target number of dimensions");
    InOutBuffersCheck bufCheck(inSize);
    if (!bufCheck.checkInputs(in.get(), out.get()))
      return Error(bufCheck.error());
    BufferAdaptor::Access outBuf(out.get());
    Result resizeResult = outBuf.resize(outSize, 1, outBuf.sampleRate());
    if (!resizeResult.ok()) return Error(BufferAlloc);
    FluidTensor<double, 1> src(inSize);
    FluidTensor<double, 1> dest(outSize);
    src = BufferAdaptor::ReadAccess(in.get()).samps(0, inSize, 0);
    mAlgorithm.transformPoint(src, dest);
    outBuf.samps(0, outSize, 0) = dest;
    return OK();
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext&)
  {
    if (!mAlgorithm.initialized()) return;
    index inSize = mAlgorithm.inputDims();
    index outSize = mAlgorithm.dims();
    if (get<kNumDimensions>() != outSize) return;
    InOutBuffersCheck bufCheck(inSize);
    if (!bufCheck.checkInputs(get<kInputBuffer>().get(),
                              get<kOutputBuffer>().get()))
      return;
    auto outBuf = BufferAdaptor::Access(get<kOutputBuffer>().get());
    if (outBuf.samps(0).size() < outSize) return;
    RealVector src(inSize);
    RealVector dest(outSize);
    src = BufferAdaptor::ReadAccess(get<kInputBuffer>().get())
              .samps(0, inSize, 0);
    mTrigger.process(input, output, [&]() {
      mAlgorithm.transformPoint(src, dest);
      outBuf.samps(0, outSize, 0) = dest;
    });
  }

  index latency() { return 0; }

  static auto getMessageDescriptors()
  {
    return defineMessages(
        makeMessage("fitTransform", &UMAPClient::fitTransform),
        makeMessage("fit", &UMAPClient::fit),
        makeMessage("transform", &UMAPClient::transform),
        makeMessage("transformPoint", &UMAPClient::transformPoint),
        makeMessage("cols", &UMAPClient::dims),
        makeMessage("clear", &UMAPClient::clear),
        makeMessage("size", &UMAPClient::size),
        makeMessage("load", &UMAPClient::load),
        makeMessage("dump", &UMAPClient::dump),
        makeMessage("write", &UMAPClient::write),
        makeMessage("read", &UMAPClient::read));
  }

private:
  FluidInputTrigger mTrigger;
};
} // namespace umap

using RTUMAPClient = ClientWrapper<umap::UMAPClient>;

} // namespace client
} // namespace fluid
