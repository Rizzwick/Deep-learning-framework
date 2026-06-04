#include <iostream>
#include <iomanip>
#include "sequential.hpp"
#include "activations.hpp"
#include "losses.hpp"

bool Tensor::grad_enabled = true;

//simple gradient-descent update so the demo stands alone without optimizers.hpp
void sgd_step(std::vector<TensorRef>& params, datatype lr){
    for(auto& p:params){
        for(size_t i=0;i<p->data.data.size();++i){
            p->data.data[i]-=lr*p->grad.data[i];
        }
    }
}

int main(){
    std::cout<<"========================================\n";
    std::cout<<"       XOR TRAINING DEMO\n";
    std::cout<<"========================================\n\n";

    //XOR truth table:inputs (4,2) and targets (4,1)
    Matrix Xdata(4,2);
    Xdata(0,0)=0; Xdata(0,1)=0;
    Xdata(1,0)=0; Xdata(1,1)=1;
    Xdata(2,0)=1; Xdata(2,1)=0;
    Xdata(3,0)=1; Xdata(3,1)=1;

    Matrix Ydata(4,1);
    Ydata(0,0)=0; Ydata(1,0)=1; Ydata(2,0)=1; Ydata(3,0)=0;

    //model: 2 -> 8 -> 1 with tanh hidden and sigmoid output
    Sequential model;
    model.add(std::make_shared<Linear>(2,8));
    model.add(std::make_shared<TanhLayer>());
    model.add(std::make_shared<Linear>(8,1));
    model.add(std::make_shared<SigmoidLayer>());

    auto params=model.parameters();

    //inputs and labels do not need gradients
    auto X=std::make_shared<Tensor>(Xdata,false);
    auto Y=std::make_shared<Tensor>(Ydata,false);

    int epochs=3000;
    datatype lr=0.1f;

    for(int epoch=0;epoch<epochs;++epoch){
        //clearing gradients before starting training
        for(auto& p:params) p->zero_grad();

        auto pred=model.forward(X);
        auto loss=mse_loss(pred,Y);

        loss->backward();
        sgd_step(params,lr);

        if(epoch%300==0){
            std::cout<<"Epoch "<<std::setw(4)<<epoch
                     <<" | Loss: "<<std::fixed<<std::setprecision(6)<<loss->data(0,0)<<"\n";
        }
    }

    //final predictions to check XOR was learned
    std::cout<<"\nFinal predictions:\n";
    auto pred=model.forward(X);
    for(size_t i=0;i<4;++i){
        std::cout<<"  Input ("<<Xdata(i,0)<<","<<Xdata(i,1)
                 <<") -> "<<std::fixed<<std::setprecision(4)<<pred->data(i,0)
                 <<"  (target "<<Ydata(i,0)<<")\n";
    }

    std::cout<<"\n========================================\n";
    return 0;
}
