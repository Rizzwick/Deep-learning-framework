#pragma once
#include <iostream>
#include <random>
#include <cassert>
#include <functional>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
typedef float datatype;

class Matrix{
    public:
    size_t rows;
    size_t cols;
    std::vector<datatype> data;

    Matrix(size_t r, size_t c): rows(r), cols(c), data(r*c,0.0f){};

    //for easy indexing we are overloading operator()
    datatype& operator()(size_t r, size_t c){
        return data[r*cols+c];
    }

    const datatype& operator()(size_t r, size_t c) const{
        return data[r*cols+c];
    }

    // weight initialization
    void randomize(datatype min=-0.1f, datatype max=0.1f){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<datatype> dis(min,max);
        for(auto& val:data){
            val=dis(gen);
        }
    }

    Matrix operator+(const Matrix& other) const{
        assert(cols==other.cols && rows==other.rows && "Dimensions must match for addition");
        Matrix result(rows,cols);
        #pragma omp parallel for
        for(long long i=0;i<(long long)data.size();i++){
            result.data[i]=data[i]+other.data[i];
        }
        return result;
    }

    Matrix operator-(const Matrix& other) const{
        assert(cols==other.cols && rows==other.rows && "Dimensions must match for subtraction");
        Matrix result(rows,cols);
        #pragma omp parallel for
        for(long long i=0;i<(long long)data.size();i++){
            result.data[i]=data[i]-other.data[i];
        }
        return result;
    }

    Matrix operator*(const datatype scalar) const{
        Matrix result(rows,cols);
        #pragma omp parallel for
        for(long long i=0;i<(long long)data.size();i++){
            result.data[i]=scalar*data[i];
        }
        return result;
    }

    Matrix matmul(const Matrix& other) const{
        assert(cols==other.rows && "Inner dimensions must match for matrix multiplication");
        Matrix result(rows,other.cols);

        //parallelize across output rows (each row writes a disjoint slice)
        #pragma omp parallel for
        for(long long i=0;i<(long long)rows;++i){
            for(size_t k=0;k<cols;++k){
                datatype temp=(*this)(i,k);
                for(size_t j=0;j<other.cols;++j){
                    result(i,j)+=temp*other(k,j);
                }
            }
        }

        return result;
    }

    Matrix transpose() const{
        Matrix result(cols,rows);
        for(size_t r=0;r<rows;++r){
            for(size_t c=0;c<cols;++c){
                result(c,r)=(*this)(r,c);
            }
        }
        return result;

    }

    void print() const{
        std::cout<<"Matrix shape ("<<rows<<", "<<cols<<")"<<std::endl;

        for(size_t i=0;i<rows;++i){
            for(size_t j=0;j<cols;++j){
                std::cout<<(*this)(i,j)<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }

    Matrix element_mul(const Matrix& other)const{
        assert(rows==other.rows && cols==other.cols && "Dimensions must match");
        Matrix result(rows,cols);
        #pragma omp parallel for
        for(long long i=0;i<(long long)data.size();++i){
            result.data[i]=other.data[i]*data[i];
        }
        return result;
    }

    Matrix map(const std::function<datatype(datatype)>& func)const{
        Matrix result(rows,cols);
        for(size_t i=0;i<data.size();++i){
            result.data[i]=func(data[i]);
        }
        return result;
    }
};