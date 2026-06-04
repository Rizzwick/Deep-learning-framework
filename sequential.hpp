#pragma once
#include <vector>
#include <memory>
#include "layers.hpp"

//chaining layers together so input flows through them in order
class Sequential: public Layer{
public:
    std::vector<std::shared_ptr<Layer>> layers;

    void add(std::shared_ptr<Layer> layer){
        layers.push_back(layer);
    }

    TensorRef forward(TensorRef x) override {
        for(auto& l:layers) x=l->forward(x);
        return x;
    }

    //gathering all trainable parameters from each layer
    std::vector<TensorRef> parameters() override {
        std::vector<TensorRef> all;
        for(auto& l:layers){
            auto p=l->parameters();
            all.insert(all.end(),p.begin(),p.end());
        }
        return all;
    }
};
