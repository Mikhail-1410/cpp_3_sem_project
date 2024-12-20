#pragma once
#include "layer.hpp"
#include <cmath>

template<typename T>
class ELULayer : public Layer<T> {
private:
    T alpha_;
    std::vector<Matrix<T>> input_cache_;
public:
    ELULayer(T alpha=1.0):alpha_(alpha){}

    std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& input) override {
        if(input.size()!=1) throw std::runtime_error("ELU forward: one channel expected.");
        input_cache_=input;
        const Matrix<T>& in=input[0];
        Matrix<T> out(in.rows(),in.cols(),0);
        for(size_t i=0;i<in.rows();++i){
            for(size_t j=0;j<in.cols();++j){
                T val=in(i,j);
                if(val>0) out(i,j)=val;
                else out(i,j)=alpha_*(std::exp(val)-1);
            }
        }
        return {out};
    }

    std::vector<Matrix<T>> backward(const std::vector<Matrix<T>>& dLoss,T learning_rate,T lambda=0.0) override {
        if(dLoss.size()!=1) throw std::runtime_error("ELU backward: one channel expected.");
        const Matrix<T>& dL=dLoss[0];
        const Matrix<T>& in=input_cache_[0];
        Matrix<T> dInput(in.rows(),in.cols(),0);
        for(size_t i=0;i<in.rows();++i){
            for(size_t j=0;j<in.cols();++j){
                T val=in(i,j);
                if(val>0) dInput(i,j)=dL(i,j);
                else dInput(i,j)=dL(i,j)*alpha_*std::exp(val);
            }
        }
        return {dInput};
    }
};