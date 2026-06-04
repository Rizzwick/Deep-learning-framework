#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "tensor.hpp"

//mean squared error:scalar =mean((pred - target)^2)
inline TensorRef mse_loss(TensorRef pred, TensorRef target){
    size_t N=pred->data.data.size();
    Matrix diff=pred->data-target->data;

    datatype loss_val=0.0f;
    for(auto v:diff.data) loss_val+=v*v;
    loss_val/=(datatype)N;

    Matrix loss_mat(1,1);
    loss_mat(0,0)=loss_val;

    bool req_grad=pred->requires_grad;
    auto out=std::make_shared<Tensor>(loss_mat,req_grad);

    if(req_grad){
        out->parents.push_back(pred);
        //save diff for the backward pass
        auto diff_ptr=std::make_shared<Matrix>(diff);
        out->backward_func=[pred,diff_ptr,out,N](){
            if(pred->requires_grad){
                //d/dpred of MSE = 2*(pred - target)/N
                datatype scale=2.0f*out->grad(0,0)/(datatype)N;
                Matrix grad_in=(*diff_ptr)*scale;
                pred->grad=pred->grad+grad_in;
            }
        };
    }
    return out;
}

//fusing softmax+negative log likelihood for numerical stability
//logits:(N,C),targets:class indices of size N
inline TensorRef softmax_cross_entropy(TensorRef logits, const std::vector<int>& targets){
    size_t N=logits->data.rows;
    size_t C=logits->data.cols;

    Matrix probs(N,C);
    datatype loss_val=0.0f;

    for(size_t i=0;i<N;++i){
        //subtracting max for stability before exponentiating
        datatype maxv=logits->data(i,0);
        for(size_t j=1;j<C;++j){
            if(logits->data(i,j)>maxv) maxv=logits->data(i,j);
        }
        datatype sum_exp=0.0f;
        for(size_t j=0;j<C;++j){
            probs(i,j)=std::exp(logits->data(i,j)-maxv);
            sum_exp+=probs(i,j);
        }
        for(size_t j=0;j<C;++j) probs(i,j)/=sum_exp;
        
        datatype p=std::max(probs(i,targets[i]),(datatype)1e-12f);
        loss_val+=-std::log(p);
    }
    loss_val/=(datatype)N;

    Matrix loss_mat(1,1);
    loss_mat(0,0)=loss_val;

    bool req_grad=logits->requires_grad;
    auto out=std::make_shared<Tensor>(loss_mat,req_grad);

    if(req_grad){
        out->parents.push_back(logits);
        auto probs_ptr=std::make_shared<Matrix>(probs);
        auto targets_copy=targets;
        out->backward_func=[logits,probs_ptr,targets_copy,out,N,C](){
            if(logits->requires_grad){
                //gradient is (softmax - one_hot)/N
                Matrix grad_in=*probs_ptr;
                for(size_t i=0;i<N;++i){
                    grad_in(i,targets_copy[i])-=1.0f;
                }
                datatype scale=out->grad(0,0)/(datatype)N;
                for(size_t i=0;i<grad_in.data.size();++i){
                    grad_in.data[i]*=scale;
                }
                logits->grad=logits->grad+grad_in;
            }
        };
    }
    return out;
}
