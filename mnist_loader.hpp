#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include "new_matrix.hpp"

//loads MNIST from CSV format:each row is "label,p1,p2,...,p784"
//pixels are normalized to [0,1]
inline std::pair<Matrix,std::vector<int>> load_mnist_csv(const std::string& path,size_t max_rows=0){
    std::ifstream file(path);
    if(!file.is_open()){
        std::cerr<<"Could not open MNIST file: "<<path<<"\n";
        return {Matrix(0,0),{}};
    }

    std::vector<std::vector<datatype>> rows;
    std::vector<int> labels;

    std::string line;
    //skipping a header row if present
    if(std::getline(file,line)){
        if(!line.empty() && (line[0]<'0'||line[0]>'9')){
            //header ignored
        } else {
            //if first line was actually data, rewind by re-parsing it
            std::stringstream ss(line);
            std::string cell;
            std::vector<datatype> pixels;
            bool first=true;
            int label=0;
            while(std::getline(ss,cell,',')){
                if(first){ label=std::stoi(cell); first=false; }
                else pixels.push_back(std::stof(cell)/255.0f);
            }
            if(!pixels.empty()){
                rows.push_back(pixels);
                labels.push_back(label);
            }
        }
    }

    while(std::getline(file,line)){
        if(line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<datatype> pixels;
        bool first=true;
        int label=0;
        while(std::getline(ss,cell,',')){
            if(first){ label=std::stoi(cell); first=false; }
            else pixels.push_back(std::stof(cell)/255.0f);
        }
        if(pixels.empty()) continue;
        rows.push_back(pixels);
        labels.push_back(label);
        if(max_rows>0 && rows.size()>=max_rows) break;
    }

    size_t N=rows.size();
    size_t F=N>0?rows[0].size():0;
    Matrix X(N,F);
    for(size_t i=0;i<N;++i){
        for(size_t j=0;j<F;++j){
            X(i,j)=rows[i][j];
        }
    }
    return {X,labels};
}
