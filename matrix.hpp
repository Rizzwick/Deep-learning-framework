#pragma once
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <tuple>
#include <functional>
#include <random>

template<typename Type>
class Matrix{
    size_t rows;
    size_t cols;

    public:
    std::vector<Type> data;
    std::tuple<size_t,size_t> shape;
    size_t num_ele=rows*cols;

    //constructor
    Matrix(size_t rows,size_t cols): cols(cols), rows(rows), data({}){
        data.resize(rows*cols,Type());
        shape={rows,cols};

    }
    //when no value given
    Matrix(): cols(0),rows(0),data({}){
        shape={rows,cols};
    }

    void print_shape(){
        std::cout<<"Matrix Size(["<<rows<<","<<cols<<"])"<<std::endl;
    }

    void print(){
        for(size_t r=0;r<rows;r++){
            for(size_t c=0;c<cols;c++){
                std::cout<<(*this)(r,c)<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }

    //for accessing the members of matrix we use operator overloading
    Type& operator()(size_t row,size_t col){
        return data[row*cols+col];
    }

    //matrix multiplication
    Matrix matmul(Matrix& other_mat){
        assert(cols==other_mat.rows);
        Matrix output(rows,other_mat.cols);

        for(size_t r=0;r<output.rows;r++){
            for(size_t c=0;c<output.cols;c++){
                for(size_t k=0;k<other_mat.rows;k++){
                    output(r,c)+=(*this)(r,k)*target(k,c);
                }
            }
        }
        return output;
    }

    //element wise multiplication
    Matrix elementwise_mul(Matrix& other){
        assert(shape=other.shape);
        Matrix output((*this));
        for(size_t r=0;r<output.rows;r++){
            for(size_t c=0;c<output.cols;c++){
                output(r,c)=other(r,c)*(*this)(r,c);
            }
        }
        return output;
    }
    
    // squaring a matrix
    Matrix square(){
        Matrix output((*this));
        output=elementwise_mul(output);
        return output;
    }

    //scalar multiplication
    Matrix scalar_mul(Type scalar){
        Matrix output(*this);
        for(size_t r=0;r<output.rows;r++){
            for(size_t c=0;c<output.cols;c++){
                other(r,c)=scalar*(*this)(r,c);
            }
        }
        return output;
    }

    //addition
    Matrix add(const Matrix& other)const {
        assert(shape==other.shape);
        Matrix output(rows,cols);
        for(size_t r=0;r<output.rows;r++){
            for(size_t c=0;c<output.cols;c++){
                output(r,c)=(*this)(r,c)+other(r,c);
            }
        }

        return output;
    }

    //operator overloading for addition 
    Matrix operator+(Matrix& other){
        return add(other);
    }

    Matrix sub(const Matrix& other) const {
        assert(shape==other.shape);
        Matrix output(rows,cols);
        for(size_t r=0;r<output.rows;r++){
            for(size_t c=0;c<output.cols;c++){
                output(r,c)=(*this)(r,c)-other(r,c);
            }
        }

        return output;
    }

    //operator overloading for subtraction 
    Matrix operator-(Matrix& other){
        return sub(other);
    }

    ///matrix transpose
    Matrix transpose(){
        size_t new_rows(cols),new_cols(rows);
        Matrix transposed(new_rows,new_cols);

        for(size_t r=0;r<new_rows;r++){
            for(size_t c=0;c<new_cols;c++){
                transposed(r,c)=(*this)(c,r);
            }
        }
        return transposed;

    }
    //for ease
    Matrix T(){
        return transpose();
    }

    Matrix func_apply(const std::function<Type(const Type &)> &function){
        Matrix output(*this);
        for(size_t r=0;r<rows;r++){
            for(size_t c=0;c<cols;c++){
                output(r,c)=function((*this)(r,c));
            }
        }
        return output;
    }

    Matrix& randomize(Type min,Type max){
        assert(max>min);
        
    }
};

