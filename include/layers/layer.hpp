#pragma once
#include "../utils/matrix.hpp"
#include <vector>

template<typename T>
class Layer {
public:
    virtual ~Layer()=default;
    virtual std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& input)=0;
    virtual std::vector<Matrix<T>> backward(const std::vector<Matrix<T>>& dLoss, T learning_rate, T lambda=0.0)=0;
};