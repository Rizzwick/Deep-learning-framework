#pragma once
#include <cmath>
#include "tensor.hpp"

//bias broadcasting along the batch dim: x(N,F) +b(1,F)
inline TensorRef add_bias(TensorRef x, TensorRef b){
    Matrix out_data(x->data.rows, x->data.cols);
    for(size_t i=0;i<x->data.rows;++i){
        for(size_t j=0;j<x->data.cols;++j){
            out_data(i,j)=x->data(i,j)+b->data(0,j);
        }
    }
    bool req_grad = Tensor::grad_enabled && (x->requires_grad || b->requires_grad);
    auto out=std::make_shared<Tensor>(out_data,req_grad);

    if(req_grad){
        out->parents.push_back(x);
        out->parents.push_back(b);

        Tensor* out_ptr = out.get();

        out->backward_func=[x,b,out_ptr](){
            if(x->requires_grad){
                x->grad=x->grad+out_ptr->grad;
            }
            if(b->requires_grad){
                //summing the upstream gradient across the batch dimension
                Matrix g(1,b->data.cols);
                for(size_t i=0;i<out_ptr->grad.rows;++i){
                    for(size_t j=0;j<out_ptr->grad.cols;++j){
                        g(0,j)+=out_ptr->grad(i,j);
                    }
                }
                b->grad=b->grad+g;
            }
        };
    }
    return out;
}

//relu activation
inline TensorRef relu(TensorRef x){
    Matrix out_data=x->data.map([](datatype v){return v>0.0f?v:0.0f;});
    bool req_grad = Tensor::grad_enabled && x->requires_grad;
    auto out=std::make_shared<Tensor>(out_data,req_grad);

    if(req_grad){
        out->parents.push_back(x);
        Tensor* out_ptr = out.get();

        out->backward_func=[x,out_ptr](){
            if(x->requires_grad){
                Matrix g(x->data.rows,x->data.cols);
                for(size_t i=0;i<g.data.size();++i){
                    //gradient only flows where input was positive
                    g.data[i]=x->data.data[i]>0.0f?out_ptr->grad.data[i]:0.0f;
                }
                x->grad=x->grad+g;
            }
        };
    }
    return out;
}

//sigmoid activation
inline TensorRef sigmoid(TensorRef x){
    Matrix out_data=x->data.map([](datatype v){return 1.0f/(1.0f+std::exp(-v));});
    bool req_grad = Tensor::grad_enabled && x->requires_grad;
    auto out=std::make_shared<Tensor>(out_data,req_grad);

    if(req_grad){
        out->parents.push_back(x);
        Tensor* out_ptr = out.get();
        
        out->backward_func=[x,out_ptr](){
            if(x->requires_grad){
                Matrix g(x->data.rows,x->data.cols);
                for(size_t i=0;i<g.data.size();++i){
                    //d/dx sigmoid = sig*(1-sig)
                    datatype s=out_ptr->data.data[i];
                    g.data[i]=out_ptr->grad.data[i]*s*(1.0f-s);
                }
                x->grad=x->grad+g;
            }
        };
    }
    return out;
}

//tanh activation
inline TensorRef tanh_op(TensorRef x){
    Matrix out_data=x->data.map([](datatype v){return std::tanh(v);});
    bool req_grad = Tensor::grad_enabled && x->requires_grad;
    auto out=std::make_shared<Tensor>(out_data,req_grad);

    if(req_grad){
        out->parents.push_back(x);
        Tensor* out_ptr = out.get();

        out->backward_func=[x,out_ptr](){
            if(x->requires_grad){
                Matrix g(x->data.rows,x->data.cols);
                for(size_t i=0;i<g.data.size();++i){
                    //d/dx tanh = 1 - tanh^2
                    datatype t=out_ptr->data.data[i];
                    g.data[i]=out_ptr->grad.data[i]*(1.0f-t*t);
                }
                x->grad=x->grad+g;
            }
        };
    }
    return out;
}
