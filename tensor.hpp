#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>
#include "new_matrix.hpp"

class Tensor;

//to use as a reference to tensors and manage pointers automatically to avoid memory leakage
using TensorRef=std::shared_ptr<Tensor>; 

class Tensor: public std::enable_shared_from_this<Tensor>{
    public:
    Matrix data;
    Matrix grad;
    bool requires_grad;
    static bool grad_enabled;

    std::vector<TensorRef> parents; //vector to store parents
    std::function<void()> backward_func; //this function calculates derivatives for backpropagation

    //constructor for a node in the computational graph
    Tensor(const Matrix& m, bool req_grad=false): data(m), grad(m.rows,m.cols), requires_grad(req_grad){
        backward_func=[](){};
    }

    //making the grads zero before a training loop
    void zero_grad(){
        for(size_t i=0;i<grad.data.size();++i){
            grad.data[i]=0.0f;
        }
    }

    void backward(){
        if(!requires_grad) return;

        std::vector<TensorRef> topo; // ordering nodes from output to inputs(topological sort)
        std::unordered_set<Tensor*> visited; //to avoid using the same node twice

        std::function<void(TensorRef)> build_topo=[&](TensorRef node){
            if(visited.find(node.get())==visited.end()){
                visited.insert(node.get());
                for(auto& parent_node:node->parents){
                    build_topo(parent_node);
                }
                topo.push_back(node);
            }
        };

        build_topo(shared_from_this());

        for(size_t i=0;i<grad.data.size();++i){
            grad.data[i]=1.0f;
        }

        for(auto it=topo.rbegin(); it!=topo.rend();++it){
            (*it)->backward_func();
        }
    }
};

//inline bool Tensor::grad_enabled = true;

//for addition (Z=A+B)
inline TensorRef add(TensorRef a, TensorRef b){
    Matrix out_data=a->data+b->data;
    bool req_grad = Tensor::grad_enabled && (a->requires_grad || b->requires_grad);

    auto out=std::make_shared<Tensor>(out_data,req_grad);

    if(req_grad){
        out->parents.push_back(a);
        out->parents.push_back(b);

        Tensor* out_ptr = out.get();
        out->backward_func=[a,b,out_ptr](){
            if(a->requires_grad){
                a->grad=a->grad+out_ptr->grad;
            }
            if(b->requires_grad){
                b->grad=b->grad+out_ptr->grad;
            }
        };
    }
    return out;
}

//for subtraction (Z=A-B)
inline TensorRef subtract(TensorRef a, TensorRef b){
    Matrix out_data=a->data - b->data;
    bool req_grad = Tensor::grad_enabled && (a->requires_grad || b->requires_grad);

    auto out=std::make_shared<Tensor>(out_data,req_grad);

    if(req_grad){
        out->parents.push_back(a);
        out->parents.push_back(b);

        Tensor* out_ptr = out.get();

        out->backward_func=[a,b,out_ptr](){
            if(a->requires_grad){
                a->grad=a->grad+out_ptr->grad;
            }
            if(b->requires_grad){
                b->grad=b->grad - out_ptr->grad;
            }
        };
    }
    return out;
}

//for elementwise multiplication Z=A.B
inline TensorRef multiply(TensorRef a, TensorRef b){
    Matrix out_data=a->data.element_mul(b->data);
    bool req_grad = Tensor::grad_enabled && (a->requires_grad || b->requires_grad);

    auto out=std::make_shared<Tensor>(out_data,req_grad);

    if(req_grad){
        out->parents.push_back(a);
        out->parents.push_back(b);

        Tensor* out_ptr = out.get();

        out->backward_func=[a,b,out_ptr](){
            if(a->requires_grad){
                a->grad=a->grad+ out_ptr->grad.element_mul(b->data);
            }
            if(b->requires_grad){
                b->grad=b->grad+ out_ptr->grad.element_mul(a->data);
            }
        };
    }
    return out;
}

//for matrix multiplication Z=A*B
inline TensorRef matmul(TensorRef a, TensorRef b){
    Matrix out_data=a->data.matmul(b->data);
    bool req_grad = Tensor::grad_enabled && (a->requires_grad || b->requires_grad);

    auto out=std::make_shared<Tensor>(out_data,req_grad);

    if(req_grad){
        out->parents.push_back(a);
        out->parents.push_back(b);

        Tensor* out_ptr = out.get();

        out->backward_func=[a,b,out_ptr](){
            if(a->requires_grad){
                Matrix b_T=b->data.transpose();
                a->grad=a->grad+ out_ptr->grad.matmul(b_T);
            }
            if(b->requires_grad){
                Matrix a_T=a->data.transpose();
                b->grad=b->grad+ a_T.matmul(out_ptr->grad);
            }
        };
    }
    return out;
}

