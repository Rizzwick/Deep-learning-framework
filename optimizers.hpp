#pragma once
#include <vector>
#include <cmath>
#include "tensor.hpp"

//abstract optimizer interface
class Optimizer{
public:
    std::vector<TensorRef> params;
    Optimizer(std::vector<TensorRef> p): params(std::move(p)) {}

    virtual void step()=0;

    //zeroes out gradients on every parameter before the next backward pass
    void zero_grad(){
        for(auto& p:params) p->zero_grad();
    }

    virtual ~Optimizer()=default;
};

//classic stochastic gradient descent
class SGD: public Optimizer{
public:
    datatype lr;

    SGD(std::vector<TensorRef> p, datatype learning_rate=0.01f): Optimizer(std::move(p)), lr(learning_rate) {}

    void step() override {
        for(auto& p:params){
            for(size_t i=0;i<p->data.data.size();++i){
                p->data.data[i]-=lr*p->grad.data[i];
            }
        }
    }
};

//adam: adaptive moments with bias correction
class Adam: public Optimizer{
public:
    datatype lr;
    datatype beta1;
    datatype beta2;
    datatype eps;
    int t; //timestep counter

    //first and second moment buffers,one per parameter
    std::vector<Matrix> m;
    std::vector<Matrix> v;

    Adam(std::vector<TensorRef> p, datatype learning_rate=1e-3f,datatype b1=0.9f, datatype b2=0.999f, datatype e=1e-8f) : Optimizer(std::move(p)), lr(learning_rate), beta1(b1), beta2(b2), eps(e), t(0){
        //allocating moment buffers matching each parameter
        for(auto& pa:params){
            m.emplace_back(pa->data.rows,pa->data.cols);
            v.emplace_back(pa->data.rows,pa->data.cols);
        }
    }

    void step() override {
        t++;
        datatype bc1=1.0f-std::pow(beta1,(datatype)t);
        datatype bc2=1.0f-std::pow(beta2,(datatype)t);

        for(size_t k=0;k<params.size();++k){
            auto& p=params[k];
            for(size_t i=0;i<p->data.data.size();++i){
                datatype g=p->grad.data[i];
                //running mean of gradients
                m[k].data[i]=beta1*m[k].data[i]+(1.0f-beta1)*g;
                //running mean of squared gradients
                v[k].data[i]=beta2*v[k].data[i]+(1.0f-beta2)*g*g;

                datatype mhat=m[k].data[i]/bc1;
                datatype vhat=v[k].data[i]/bc2;

                p->data.data[i]-=lr*mhat/(std::sqrt(vhat)+eps);
            }
        }
    }
};
