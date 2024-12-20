#pragma once
#include "layer.hpp"
#include <cmath>
#include <random>

template<typename T>
class FullyConnectedLayer : public Layer<T> {
private:
    Matrix<T> weights_;
    Matrix<T> biases_;
    std::vector<Matrix<T>> input_cache_;
public:
    FullyConnectedLayer(int input_size,int output_size)
        : weights_(input_size,output_size,0), biases_(1,output_size,0) {
        initialize_weights();
    }

    std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& input) override {
        if(input.size()!=1) throw std::runtime_error("FCL forward: expected one channel.");
        const Matrix<T>& in=input[0];
        input_cache_=input;

        Matrix<T> output(in.rows(),weights_.cols(),0);
        for(size_t i=0;i<in.rows();++i){
            for(size_t j=0;j<weights_.cols();++j){
                T sum=0;
                for(size_t k=0;k<weights_.rows();++k){
                    sum+=in(i,k)*weights_(k,j);
                }
                output(i,j)=sum+biases_(0,j);
            }
        }

        return {output};
    }

    std::vector<Matrix<T>> backward(const std::vector<Matrix<T>>& dLoss, T learning_rate, T lambda=0.0) override {
        if(dLoss.size()!=1) throw std::runtime_error("FCL backward: one channel expected.");
        const Matrix<T>& dL=dLoss[0];
        const Matrix<T>& in=input_cache_[0];

        if(dL.rows()!=in.rows()) throw std::runtime_error("FCL backward: dim mismatch");

        Matrix<T> dWeights(weights_.rows(),weights_.cols(),0);
        Matrix<T> dBiases(1,weights_.cols(),0);
        for(size_t i=0;i<in.rows();++i){
            for(size_t j=0;j<weights_.cols();++j){
                dBiases(0,j)+=dL(i,j);
                for(size_t k=0;k<weights_.rows();++k){
                    dWeights(k,j)+=in(i,k)*dL(i,j);
                }
            }
        }

        if(lambda>0){
            for(size_t i=0;i<weights_.rows();++i){
                for(size_t j=0;j<weights_.cols();++j){
                    dWeights(i,j)+=lambda*weights_(i,j);
                }
            }
        }

        for(size_t i=0;i<weights_.rows();++i){
            for(size_t j=0;j<weights_.cols();++j){
                weights_(i,j)-=learning_rate*dWeights(i,j);
            }
        }

        for(size_t j=0;j<biases_.cols();++j){
            biases_(0,j)-=learning_rate*dBiases(0,j);
        }

        Matrix<T> dInput(in.rows(),weights_.rows(),0);
        for(size_t i=0;i<dL.rows();++i){
            for(size_t j=0;j<weights_.rows();++j){
                for(size_t k=0;k<weights_.cols();++k){
                    dInput(i,j)+=dL(i,k)*weights_(j,k);
                }
            }
        }

        return {dInput};
    }

private:
    void initialize_weights(){
        std::mt19937 gen(std::random_device{}());
        T stddev=std::sqrt((T)2.0/(T)weights_.rows());
        std::normal_distribution<T> dist(0,stddev);
        for(size_t i=0;i<weights_.rows();++i){
            for(size_t j=0;j<weights_.cols();++j){
                weights_(i,j)=dist(gen);
            }
        }
        for(size_t j=0;j<biases_.cols();++j){
            biases_(0,j)=0;
        }
    }
};