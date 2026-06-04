#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <utility>
#include "tensor.hpp"

//iterates through a dataset in shuffled mini-batches
class DataLoader{
public:
    Matrix X;             //full dataset(feature) matrix(N,F)
    std::vector<int> y;   //class labels of length N
    size_t batch_size;
    bool shuffle;

    std::vector<size_t> indices; //shuffled access order
    size_t cursor;

    DataLoader(const Matrix& X_full, const std::vector<int>& y_full,size_t bs, bool sh=true): X(X_full), y(y_full),batch_size(bs), shuffle(sh), cursor(0){
        indices.resize(X.rows);
        for(size_t i=0;i<X.rows;++i) indices[i]=i;
        reset();
    }

    //reshuffle and rewind to the start of the dataset
    void reset(){
        cursor=0;
        if(shuffle){
            static std::mt19937 gen(std::random_device{}());
            std::shuffle(indices.begin(),indices.end(),gen);
        }
    }

    //true if at least one more batch is available
    bool has_next() const {
        return cursor<X.rows;
    }

    //number of batches per epoch
    size_t num_batches() const {
        return (X.rows+batch_size-1)/batch_size;
    }

    //returns the next mini-batch as (features tensor, label vector)
    std::pair<TensorRef,std::vector<int>> next_batch(){
        size_t end=std::min(cursor+batch_size,X.rows);
        size_t actual=end-cursor;

        Matrix Xb(actual,X.cols);
        std::vector<int> yb(actual);

        for(size_t i=0;i<actual;++i){
            size_t idx=indices[cursor+i];
            for(size_t j=0;j<X.cols;++j){
                Xb(i,j)=X(idx,j);
            }
            yb[i]=y[idx];
        }
        cursor=end;

        auto X_tensor=std::make_shared<Tensor>(Xb,false);
        return {X_tensor,yb};
    }
};
