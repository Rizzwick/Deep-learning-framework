#pragma once
#include <vector>
#include <memory>
#include <cmath>
#include "tensor.hpp"
#include "tensor_ops.hpp"

//abstract base class for all layers
class Layer{
public:
    virtual TensorRef forward(TensorRef x)=0;
    virtual std::vector<TensorRef> parameters(){return {};}
    virtual ~Layer()=default;
};

//fully connected layer:y =x*W+b
class Linear: public Layer{
public:
    TensorRef weights; //shape (in_features, out_features)
    TensorRef bias;    //shape (1, out_features)

    Linear(size_t in_features, size_t out_features){
        Matrix w(in_features,out_features);
        //Xavier uniform initialization for stable training
        datatype bound=std::sqrt(6.0f/(datatype)(in_features+out_features));
        w.randomize(-bound,bound);
        weights=std::make_shared<Tensor>(w,true);

        Matrix b(1,out_features);
        //bias starting at zero
        bias=std::make_shared<Tensor>(b,true);
    }

    TensorRef forward(TensorRef x) override {
        auto z=matmul(x,weights);
        return add_bias(z,bias);
    }

    std::vector<TensorRef> parameters() override {
        return {weights,bias};
    }
};
