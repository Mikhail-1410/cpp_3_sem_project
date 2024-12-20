#pragma once
#include "layer.hpp"
#include <stdexcept>

template<typename T>
class PoolingLayer : public Layer<T> {
private:
    size_t pool_size_;
    size_t stride_;
public:
    PoolingLayer(size_t pool_size=2,size_t stride=2):pool_size_(pool_size),stride_(stride){}

    std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& input) override {
        // Столько же каналов, сколько на входе
        std::vector<Matrix<T>> output;
        output.reserve(input.size());
        for(auto &ch : input){
            size_t output_height=(ch.rows()-pool_size_)/stride_+1;
            size_t output_width=(ch.cols()-pool_size_)/stride_+1;
            Matrix<T> pooled(output_height,output_width,0);
            for(size_t i=0;i<output_height;++i){
                for(size_t j=0;j<output_width;++j){
                    T max_val=ch(i*stride_,j*stride_);
                    for(size_t pi=0;pi<pool_size_;++pi){
                        for(size_t pj=0;pj<pool_size_;++pj){
                            T current=ch(i*stride_+pi,j*stride_+pj);
                            if(current>max_val) max_val=current;
                        }
                    }
                    pooled(i,j)=max_val;
                }
            }
            output.push_back(pooled);
        }
        return output;
    }

    std::vector<Matrix<T>> backward(const std::vector<Matrix<T>>& dLoss, T learning_rate, T lambda=0.0) override {
        // Не реализован обратный проход (заглушка)
        // Возвращаем нули размером как вход
        // Для корректной работы CNN обычно нужен обратный проход,
        // но сейчас можно оставить заглушку.
        // Допустим, у нас есть input_cache_ для backward, но сейчас заглушка.
        std::vector<Matrix<T>> dInput; 
        dInput.reserve(dLoss.size());
        for (auto &ch : dLoss) {
            Matrix<T> zero(ch.rows()*stride_+pool_size_-1, ch.cols()*stride_+pool_size_-1,0);
            // Заглушка, без реального backward pooling
            dInput.push_back(zero);
        }
        return dInput;
    }
};