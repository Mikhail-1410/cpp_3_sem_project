#pragma once
#include "network.hpp"
#include "utils/logger.hpp"
#include "utils/metrics.hpp"
#include "exception.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <tuple>
#include <limits>

enum class LossFunction {
    MSE,
    CrossEntropy,
    Hinge
};

template<typename T>
class Trainer {
public:
    static T mse_loss(const Matrix<T>& pred, const Matrix<T>& target) {
        if(pred.rows()!=target.rows()||pred.cols()!=target.cols()) throw std::runtime_error("MSE: dim mismatch");
        T sum=0;
        size_t count=pred.rows()*pred.cols();
        for(size_t i=0;i<pred.rows();++i){
            for(size_t j=0;j<pred.cols();++j){
                T diff=pred(i,j)-target(i,j);
                sum+=diff*diff;
            }
        }
        return sum/(T)count;
    }

    static Matrix<T> mse_loss_grad(const Matrix<T>& pred, const Matrix<T>& target){
        if(pred.rows()!=target.rows()||pred.cols()!=target.cols()) throw std::runtime_error("MSE_grad: dim mismatch");
        Matrix<T> grad(pred.rows(), pred.cols(),0);
        size_t count=pred.rows()*pred.cols();
        for(size_t i=0;i<pred.rows();++i){
            for(size_t j=0;j<pred.cols();++j){
                grad(i,j)=2*(pred(i,j)-target(i,j))/ (T)count;
            }
        }
        return grad;
    }

    static T cross_entropy_loss(const Matrix<T>& pred, const Matrix<T>& target){
        if(pred.rows()!=target.rows()||pred.cols()!=target.cols()) throw std::runtime_error("CE: dim mismatch");
        T loss=0;
        for(size_t i=0;i<pred.rows();++i){
            for(size_t j=0;j<pred.cols();++j){
                loss-=target(i,j)*std::log(pred(i,j)+(T)1e-15);
            }
        }
        return loss/(T)pred.rows();
    }

    static Matrix<T> cross_entropy_loss_grad(const Matrix<T>& pred, const Matrix<T>& target){
        if(pred.rows()!=target.rows()||pred.cols()!=target.cols()) throw std::runtime_error("CE_grad: dim mismatch");
        Matrix<T> grad(pred.rows(), pred.cols(),0);
        for(size_t i=0;i<pred.rows();++i){
            for(size_t j=0;j<pred.cols();++j){
                grad(i,j)=-target(i,j)/(pred(i,j)+(T)1e-15);
            }
        }
        return grad;
    }

    std::tuple<T,float,float,float, T,float,float,float> train(Network<T>& net, 
                                          const std::vector<Matrix<T>>& X,
                                          const std::vector<Matrix<T>>& Y,
                                          size_t epochs, T learning_rate, size_t batch_size=32,
                                          T lambda=0.0, size_t patience=10, T min_delta=1e-4,
                                          LossFunction loss_fn=LossFunction::MSE) {
        if(X.size()!=1||Y.size()!=1) throw std::runtime_error("Trainer: Expect single matrix for X and Y");
        const Matrix<T>& X_full = X[0];
        const Matrix<T>& Y_full = Y[0];
        size_t num_samples = X_full.rows();
        if(num_samples == 0) throw std::runtime_error("No data");
        size_t feature_dim = X_full.cols();
        size_t num_classes = Y_full.cols();
        size_t num_batches = num_samples/batch_size;

        T best_loss=std::numeric_limits<T>::max();
        size_t wait=0;
        T final_train_loss=0;
        float final_train_acc=0.0f, final_train_f1=0.0f, final_train_auc=0.0f;
        float final_val_loss=0.0f, final_val_acc=0.0f, final_val_f1=0.0f, final_val_auc=0.0f;

        std::vector<size_t> indices(num_samples);
        for(size_t i=0;i<num_samples;++i) indices[i]=i;

        for(size_t epoch=0;epoch<epochs;++epoch){
            try{
                std::shuffle(indices.begin(),indices.end(),std::mt19937(std::random_device{}()));

                T epoch_loss=0;
                float sum_train_acc=0.0f,sum_train_f1=0.0f,sum_train_auc=0.0f;

                for(size_t batch=0;batch<num_batches;++batch){
                    size_t start=batch*batch_size;
                    size_t end=start+batch_size;
                    Matrix<T> X_batch(batch_size, feature_dim,0);
                    Matrix<T> Y_batch(batch_size, num_classes,0);
                    for(size_t i=start;i<end;++i){
                        size_t bi=i-start;
                        for(size_t j=0;j<feature_dim;++j){
                            X_batch(bi,j)=X_full(indices[i],j);
                        }
                        for(size_t c=0;c<num_classes;++c){
                            Y_batch(bi,c)=Y_full(indices[i],c);
                        }
                    }

                    auto preds = net.forward({X_batch});
                    if(preds.size()!=1) throw std::runtime_error("Trainer::train: Network output should have one channel");
                    const Matrix<T>& predictions = preds[0];

                    T loss;
                    Matrix<T> grad;
                    if(loss_fn==LossFunction::MSE){
                        loss=mse_loss(predictions,Y_batch);
                        grad=mse_loss_grad(predictions,Y_batch);
                    } else if(loss_fn==LossFunction::CrossEntropy){
                        loss=cross_entropy_loss(predictions,Y_batch);
                        grad=cross_entropy_loss_grad(predictions,Y_batch);
                    } else {
                        throw std::runtime_error("Hinge loss not implemented");
                    }

                    epoch_loss+=loss;
                    float acc=Metrics<T>::accuracy(predictions,Y_batch);
                    float f1=Metrics<T>::f1_score(predictions,Y_batch,num_classes);
                    float auc=Metrics<T>::roc_auc_multiclass(predictions,Y_batch,num_classes);

                    sum_train_acc+=acc;
                    sum_train_f1+=f1;
                    sum_train_auc+=auc;

                    net.backward({grad},learning_rate,lambda);
                }

                T epoch_loss_avg=epoch_loss/(T)num_batches;
                float train_acc_avg=sum_train_acc/(float)num_batches;
                float train_f1_avg=sum_train_f1/(float)num_batches;
                float train_auc_avg=sum_train_auc/(float)num_batches;

                // Оценка на полном наборе
                auto pred_full_vec = net.forward({X_full});
                if(pred_full_vec.size()!=1) throw std::runtime_error("Full dataset prediction not single channel");
                const Matrix<T>& pred_full=pred_full_vec[0];

                T val_loss;
                if(loss_fn==LossFunction::MSE){
                    val_loss=mse_loss(pred_full,Y_full);
                } else if(loss_fn==LossFunction::CrossEntropy){
                    val_loss=cross_entropy_loss(pred_full,Y_full);
                } else {
                    throw std::runtime_error("Hinge not implemented for val_loss");
                }

                float val_acc=Metrics<T>::accuracy(pred_full,Y_full);
                float val_f1=Metrics<T>::f1_score(pred_full,Y_full,num_classes);
                float val_auc=Metrics<T>::roc_auc_multiclass(pred_full,Y_full,num_classes);

                Logger::log_metrics(epoch,
                                    epoch_loss_avg, train_acc_avg, train_f1_avg, train_auc_avg,
                                    val_loss, val_acc, val_f1, val_auc);

                if(epoch_loss_avg+min_delta<best_loss){
                    best_loss=epoch_loss_avg;
                    wait=0;
                } else {
                    wait++;
                    if(wait>=patience){
                        Logger::info("Early stopping on epoch "+std::to_string(epoch)+" with loss "+std::to_string(epoch_loss_avg));
                        final_train_loss=epoch_loss_avg;
                        final_train_acc=train_acc_avg;
                        final_train_f1=train_f1_avg;
                        final_train_auc=train_auc_avg;
                        final_val_loss=val_loss;
                        final_val_acc=val_acc;
                        final_val_f1=val_f1;
                        final_val_auc=val_auc;
                        break;
                    }
                }

                if(epoch%10==0){
                    Logger::info("Epoch "+std::to_string(epoch)+" - Train Loss: "+std::to_string(epoch_loss_avg)+
                                 ", Train Acc: "+std::to_string(train_acc_avg)+
                                 ", Val Loss: "+std::to_string(val_loss)+
                                 ", Val Acc: "+std::to_string(val_acc));
                }

                final_train_loss=epoch_loss_avg;
                final_train_acc=train_acc_avg;
                final_train_f1=train_f1_avg;
                final_train_auc=train_auc_avg;
                final_val_loss=val_loss;
                final_val_acc=val_acc;
                final_val_f1=val_f1;
                final_val_auc=val_auc;

            }catch(const std::exception &ex){
                Logger::error(std::string("Exception during training epoch ")+std::to_string(epoch)+": "+ex.what());
                break;
            }
        }

        return std::make_tuple(final_train_loss, final_train_acc, final_train_f1, final_train_auc,
                               final_val_loss, final_val_acc, final_val_f1, final_val_auc);
    }
};