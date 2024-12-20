#pragma once
#include <vector>
#include <stdexcept>
#include <iostream>

template<typename T>
class Matrix {
private:
    size_t rows_;
    size_t cols_;
    std::vector<T> data_;
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(size_t rows, size_t cols, T val=T()) : rows_(rows), cols_(cols), data_(rows*cols,val) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    T& operator()(size_t r, size_t c) {
        if(r>=rows_||c>=cols_) throw std::out_of_range("Matrix index out of range");
        return data_[r*cols_+c];
    }
    const T& operator()(size_t r, size_t c) const {
        if(r>=rows_||c>=cols_) throw std::out_of_range("Matrix index out of range");
        return data_[r*cols_+c];
    }

    void print() const {
        for(size_t i=0;i<rows_;++i){
            for(size_t j=0;j<cols_;++j){
                std::cout<<(*this)(i,j)<<" ";
            }
            std::cout<<"\n";
        }
    }
};