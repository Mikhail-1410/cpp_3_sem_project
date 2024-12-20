#pragma once
#include "layer.hpp"
#include <cmath>

template<typename T>
class SoftmaxLayer : public Layer<T> {
private:
    std::vector<Matrix<T>> input_cache_;
    Matrix<T> output_cache_;
public:
    SoftmaxLayer(){}

    std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& input) override {
        if(input.size()!=1) throw std::runtime_error("Softmax forward: one channel expected.");
        const Matrix<T>& in=input[0];
        input_cache_=input;
        output_cache_=in;
        for(size_t i=0;i<in.rows();++i){
            T max_val=in(i,0);
            for(size_t j=1;j<in.cols();++j){
                if(in(i,j)>max_val) max_val=in(i,j);
            }
            T sum=0;
            for(size_t j=0;j<in.cols();++j){
                output_cache_(i,j)=std::exp(in(i,j)-max_val);
                sum+=output_cache_(i,j);
            }
            for(size_t j=0;j<in.cols();++j){
                output_cache_(i,j)/=sum;
            }
        }
        return {output_cache_};
    }

    std::vector<Matrix<T>> backward(const std::vector<Matrix<T>>& dLoss,T learning_rate,T lambda=0.0) override {
        if(dLoss.size()!=1) throw std::runtime_error("Softmax backward: one channel expected.");
        const Matrix<T>& dL=dLoss[0];
        const Matrix<T>& in=input_cache_[0];
        if(dL.rows()!=in.rows()||dL.cols()!=in.cols()) throw std::runtime_error("Softmax backward: dim mismatch");

        // Предполагаем dLoss уже учитывает Softmax+CE
        // Тогда dInput=dL напрямую
        Matrix<T> dInput=dL;
        return {dInput};
    }
};