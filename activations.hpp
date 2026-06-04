#pragma once
#include "layers.hpp"
#include "tensor_ops.hpp"

//relu  as a layer
class ReLULayer: public Layer{
public:
    TensorRef forward(TensorRef x) override { return relu(x); }
};

//sigmoid as a layer
class SigmoidLayer: public Layer{
public:
    TensorRef forward(TensorRef x) override { return sigmoid(x); }
};

//tanh as a layer
class TanhLayer: public Layer{
public:
    TensorRef forward(TensorRef x) override { return tanh_op(x); }
};
