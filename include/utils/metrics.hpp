#pragma once
#include "matrix.hpp"
#include <cmath>
#include <stdexcept>

template<typename T>
class Metrics {
public:
    static size_t argmax(const Matrix<T>& mat, size_t row) {
        if(mat.rows()==0||mat.cols()==0) throw std::runtime_error("argmax on empty matrix");
        size_t max_idx=0;
        T max_val=mat(row,0);
        for(size_t j=1;j<mat.cols();++j){
            if(mat(row,j)>max_val){
                max_val=mat(row,j);
                max_idx=j;
            }
        }
        return max_idx;
    }

    static float accuracy(const Matrix<T>& predictions,const Matrix<T>& targets) {
        if(predictions.rows()!=targets.rows()||predictions.cols()!=targets.cols())
            throw std::runtime_error("Accuracy: dim mismatch");
        size_t correct=0;
        for(size_t i=0;i<predictions.rows();++i){
            size_t pred_class=argmax(predictions,i);
            size_t true_class=argmax(targets,i);
            if(pred_class==true_class) correct++;
        }
        return (float)correct/(float)predictions.rows();
    }

    static float f1_score(const Matrix<T>& predictions,const Matrix<T>& targets,size_t num_classes) {
        // Упрощённый micro-F1
        if(predictions.rows()!=targets.rows()||predictions.cols()!=targets.cols())
            throw std::runtime_error("F1: dim mismatch");
        size_t tp=0,fp=0,fn=0;
        for(size_t i=0;i<predictions.rows();++i){
            size_t pred_class=argmax(predictions,i);
            size_t true_class=argmax(targets,i);
            if(pred_class==true_class) tp++;
            else {
                fp++;
                fn++;
            }
        }
        float precision=(tp+fp==0)?0.0f:(float)tp/(tp+fp);
        float recall=(tp+fn==0)?0.0f:(float)tp/(tp+fn);
        if(precision+recall==0) return 0.0f;
        return 2.0f*(precision*recall)/(precision+recall);
    }

    static float roc_auc_multiclass(const Matrix<T>& predictions,const Matrix<T>& targets,size_t num_classes) {
        // Заглушка
        return 0.5f;
    }
};