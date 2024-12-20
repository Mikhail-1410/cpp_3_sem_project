#pragma once
#include "layer.hpp"
#include <stdexcept>

/**
 * FlattenLayer: преобразует несколько каналов [C x H x W] в один канал [N x (C*H*W)]
 * Предполагается, что N - количество образцов (строк), а C,H,W - каналы и размер.
 * Здесь N это rows, а мы объединяем каналы построчно.
 */
template<typename T>
class FlattenLayer : public Layer<T> {
public:
    std::vector<Matrix<T>> input_cache_;
    FlattenLayer(){}

    std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& input) override {
        // Все каналы имеют одинаковый размер
        size_t c=input.size();
        if(c==0) throw std::runtime_error("Flatten forward: no input channels");
        size_t rows=input[0].rows();
        size_t cols=input[0].cols();

        // Выход: одна матрица [rows x (c*cols)]
        // Считаем, что размерность по batch - это rows, мы просто склеиваем каналы по горизонтали
        size_t total_cols=c*cols;
        Matrix<T> out(rows,total_cols,0);

        for(size_t channel=0;channel<c;++channel){
            for(size_t i=0;i<rows;++i){
                for(size_t j=0;j<cols;++j){
                    out(i, channel*cols+j)=input[channel](i,j);
                }
            }
        }

        input_cache_=input;
        return {out};
    }

    std::vector<Matrix<T>> backward(const std::vector<Matrix<T>>& dLoss,T learning_rate,T lambda=0.0) override {
        if(dLoss.size()!=1) throw std::runtime_error("Flatten backward: one channel expected.");
        const Matrix<T>& dL=dLoss[0];
        // Восстанавливаем каналы
        size_t c=input_cache_.size();
        size_t rows=input_cache_[0].rows();
        size_t cols=input_cache_[0].cols();

        if(dL.rows()!=rows||dL.cols()!=c*cols) throw std::runtime_error("Flatten backward: dim mismatch");

        std::vector<Matrix<T>> dInput;
        dInput.reserve(c);
        for(size_t channel=0;channel<c;++channel){
            Matrix<T> ch(rows,cols,0);
            for(size_t i=0;i<rows;++i){
                for(size_t j=0;j<cols;++j){
                    ch(i,j)=dL(i,channel*cols+j);
                }
            }
            dInput.push_back(ch);
        }
        return dInput;
    }
};