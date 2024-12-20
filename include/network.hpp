#pragma once
#include <vector>
#include <memory>
#include "layers/layer.hpp"

template<typename T>
class Network {
private:
    std::vector<std::unique_ptr<Layer<T>>> layers_;
public:
    Network()=default;

    void add_layer(std::unique_ptr<Layer<T>> layer){
        layers_.emplace_back(std::move(layer));
    }

    std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& input){
        std::vector<Matrix<T>> current_input=input;
        for(auto &layer: layers_){
            current_input=layer->forward(current_input);
        }
        return current_input;
    }

    void backward(const std::vector<Matrix<T>>& dLoss,T learning_rate,T lambda=0.0){
        std::vector<Matrix<T>> grad=dLoss;
        for(auto it=layers_.rbegin(); it!=layers_.rend();++it){
            grad=(*it)->backward(grad,learning_rate,lambda);
        }
    }
};