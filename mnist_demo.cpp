#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <string>
#include <cstdint>
#include <cstdlib>
#include "tensor.hpp"
#include "sequential.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include "optimizers.hpp"
#include "dataloader.hpp"
#include "mnist_loader.hpp"

bool Tensor::grad_enabled = true;

//computes top-1 accuracy on the given dataset
datatype evaluate(Sequential& model, const Matrix& X, const std::vector<int>& y,
                  size_t batch_size=256){
    size_t correct=0;
    size_t N=X.rows;
    //disable graph building during eval for speed + memory
    bool prev=Tensor::grad_enabled;
    Tensor::grad_enabled=false;
    for(size_t start=0;start<N;start+=batch_size){
        size_t end=std::min(start+batch_size,N);
        size_t bs=end-start;
        Matrix Xb(bs,X.cols);
        for(size_t i=0;i<bs;++i){
            for(size_t j=0;j<X.cols;++j) Xb(i,j)=X(start+i,j);
        }
        auto Xt=std::make_shared<Tensor>(Xb,false);
        auto out=model.forward(Xt);
        for(size_t i=0;i<bs;++i){
            size_t best=0;
            datatype best_val=out->data(i,0);
            for(size_t j=1;j<(size_t)out->data.cols;++j){
                if(out->data(i,j)>best_val){ best_val=out->data(i,j); best=j; }
            }
            if((int)best==y[start+i]) correct++;
        }
    }
    Tensor::grad_enabled=prev;
    return (datatype)correct/(datatype)N;
}

void save_model(const std::string& path, std::vector<TensorRef> params){
    std::ofstream f(path,std::ios::binary);
    if(!f){ std::cerr<<"Could not open "<<path<<" for writing\n"; return; }
    int32_t n=(int32_t)params.size();
    f.write(reinterpret_cast<const char*>(&n),sizeof(n));
    for(auto& p:params){
        int32_t r=(int32_t)p->data.rows, c=(int32_t)p->data.cols;
        f.write(reinterpret_cast<const char*>(&r),sizeof(r));
        f.write(reinterpret_cast<const char*>(&c),sizeof(c));
        f.write(reinterpret_cast<const char*>(p->data.data.data()),
                sizeof(datatype)*r*c);
    }
    std::cout<<"Saved model to "<<path<<" ("<<n<<" tensors)\n";
}

int main(int argc, char** argv){
    std::cout<<"========================================\n";
    std::cout<<" MNIST TRAINING\n";
    std::cout<<"========================================\n\n";

    //defaults — tuned for fast training
    std::string train_path="data/mnist_train.csv";
    std::string test_path ="data/mnist_test.csv";
    std::string model_path="mnist_model.bin";
    int epochs=3;
    size_t batch_size=128;
    size_t max_train=0; //0 = use all

    //very small flag parser
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        auto next=[&](const char* name)->std::string{
            if(i+1>=argc){ std::cerr<<name<<" needs a value\n"; std::exit(1); }
            return argv[++i];
        };
        if(a=="--train") train_path=next("--train");
        else if(a=="--test") test_path=next("--test");
        else if(a=="--model") model_path=next("--model");
        else if(a=="--epochs") epochs=std::stoi(next("--epochs"));
        else if(a=="--batch") batch_size=(size_t)std::stoi(next("--batch"));
        else if(a=="--max-train") max_train=(size_t)std::stoi(next("--max-train"));
        else if(a=="--help"){
            std::cout<<"Usage: mnist_demo.exe [--train PATH] [--test PATH] [--model PATH]\n"
                     <<"                       [--epochs N] [--batch N] [--max-train N]\n";
            return 0;
        }
        else if(a[0]!='-'){
            if(train_path=="data/mnist_train.csv") train_path=a;
            else if(test_path=="data/mnist_test.csv") test_path=a;
        }
    }

    std::cout<<"Config: epochs="<<epochs<<" batch="<<batch_size
             <<(max_train?(" max_train="+std::to_string(max_train)):"")<<"\n";

    std::cout<<"Loading training data from "<<train_path<<"...\n";
    auto train_set=load_mnist_csv(train_path,max_train);
    Matrix& X_train=train_set.first;
    std::vector<int>& y_train=train_set.second;
    std::cout<<"Loading test data from "<<test_path<<"...\n";
    auto test_set=load_mnist_csv(test_path);
    Matrix& X_test=test_set.first;
    std::vector<int>& y_test=test_set.second;

    if(X_train.rows==0){
        std::cerr<<"No training data found. Place MNIST CSV files in ./data/\n";
        return 1;
    }
    std::cout<<"Train: "<<X_train.rows<<" samples, Test: "<<X_test.rows<<" samples\n\n";

    //mlp 784 -> 128 -> 64 -> 10
    Sequential model;
    model.add(std::make_shared<Linear>(784,128));
    model.add(std::make_shared<ReLULayer>());
    model.add(std::make_shared<Linear>(128,64));
    model.add(std::make_shared<ReLULayer>());
    model.add(std::make_shared<Linear>(64,10));

    Adam opt(model.parameters(),1e-3f);

    DataLoader loader(X_train,y_train,batch_size,/*shuffle*/true);

    for(int epoch=1;epoch<=epochs;++epoch){
        auto t0=std::chrono::high_resolution_clock::now();
        loader.reset();
        datatype running_loss=0.0f;
        size_t batches=0;

        while(loader.has_next()){
            auto batch=loader.next_batch();
            TensorRef Xb=batch.first;
            std::vector<int>& yb=batch.second;
            opt.zero_grad();
            auto logits=model.forward(Xb);
            auto loss=softmax_cross_entropy(logits,yb);
            loss->backward();
            opt.step();
            running_loss+=loss->data(0,0);
            batches++;
        }

        auto t1=std::chrono::high_resolution_clock::now();
        double secs=std::chrono::duration<double>(t1-t0).count();
        datatype avg_loss=running_loss/(datatype)batches;

        datatype test_acc=X_test.rows>0?evaluate(model,X_test,y_test):0.0f;
        std::cout<<"Epoch "<<epoch
                 <<" | Loss: "<<std::fixed<<std::setprecision(4)<<avg_loss
                 <<" | Test Acc: "<<std::setprecision(2)<<(test_acc*100.0f)<<"%"
                 <<" | Time: "<<std::setprecision(2)<<secs<<"s\n";
    }

    save_model(model_path,model.parameters());

    std::cout<<"\n========================================\n";
    return 0;
}
