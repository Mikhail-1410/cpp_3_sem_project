#include <iostream>
#include "../include/network.hpp"
#include "../include/layers/convolutional_layer.hpp"
#include "../include/layers/pooling_layer.hpp"
#include "../include/layers/fully_connected_layer.hpp"
#include "../include/layers/leaky_relu_layer.hpp"
#include "../include/layers/elu_layer.hpp"
#include "../include/layers/softmax_layer.hpp"
#include "../include/layers/flatten_layer.hpp"
#include "../include/utils/dataset.hpp"
#include "../include/utils/logger.hpp"
#include "../include/utils/metrics.hpp"
#include "../include/utils/cross_validation.hpp"
#include "../include/trainer.hpp"

int main() {
    Logger::init("training_metrics.csv");

    try {
        using T=float;

        std::string images_path="../data/mnist/train-images-idx3-ubyte";
        std::string labels_path="../data/mnist/train-labels-idx1-ubyte";
        std::vector<MNISTImage> dataset=MNISTDataset::load_mnist(images_path,labels_path);

        size_t k=5;
        auto folds=CrossValidator::k_fold_split(dataset,k);

        float total_accuracy=0.0f,total_f1=0.0f,total_auc=0.0f;
        size_t fold_num=1;

        for(auto &fold: folds){
            std::vector<MNISTImage> training_data=fold.first;
            std::vector<MNISTImage> validation_data=fold.second;

            size_t num_features=28*28;
            size_t num_classes=10;
            size_t training_size=training_data.size();
            size_t validation_size=validation_data.size();

            Matrix<T> X_train_mat(training_size,num_features,0);
            Matrix<T> Y_train_mat(training_size,num_classes,0);
            for(size_t i=0;i<training_size;++i){
                for(size_t j=0;j<num_features;++j){
                    X_train_mat(i,j)=training_data[i].pixels(j/28,j%28);
                }
                Y_train_mat(i,training_data[i].label)=1;
            }

            Matrix<T> X_val_mat(validation_size,num_features,0);
            Matrix<T> Y_val_mat(validation_size,num_classes,0);
            for(size_t i=0;i<validation_size;++i){
                for(size_t j=0;j<num_features;++j){
                    X_val_mat(i,j)=validation_data[i].pixels(j/28,j%28);
                }
                Y_val_mat(i,validation_data[i].label)=1;
            }

            Network<T> net;
            // CNN: 1->8 channels
            net.add_layer(std::make_unique<ConvolutionalLayer<T>>(1,8,3,1,1));
            net.add_layer(std::make_unique<PoolingLayer<T>>(2,2));
            // 8->16 channels
            net.add_layer(std::make_unique<ConvolutionalLayer<T>>(8,16,3,1,1));
            net.add_layer(std::make_unique<PoolingLayer<T>>(2,2));
            // Flatten -> FullyConnected -> ELU -> FullyConnected -> Softmax
            net.add_layer(std::make_unique<FlattenLayer<T>>());
            net.add_layer(std::make_unique<FullyConnectedLayer<T>>(7*7*16,128));
            net.add_layer(std::make_unique<ELULayer<T>>());
            net.add_layer(std::make_unique<FullyConnectedLayer<T>>(128,num_classes));
            net.add_layer(std::make_unique<SoftmaxLayer<T>>());

            size_t epochs=20;
            T learning_rate=0.001f;
            size_t batch_size=64;
            T lambda=0.0001f;
            size_t patience=5;
            T min_delta=1e-4f;
            LossFunction loss_fn=LossFunction::CrossEntropy;

            Trainer<T> trainer;
            auto [train_loss,train_acc,train_f1,train_auc,
                  val_loss,val_acc,val_f1,val_auc]=
                  trainer.train(net,{X_train_mat},{Y_train_mat},epochs,learning_rate,batch_size,lambda,patience,min_delta,loss_fn);

            std::cout<<"Fold "<<fold_num<<":\n";
            std::cout<<"Train Loss: "<<train_loss<<"\n";
            std::cout<<"Train Accuracy: "<<train_acc<<"\n";
            std::cout<<"Train F1 Score: "<<train_f1<<"\n";
            std::cout<<"Train ROC AUC: "<<train_auc<<"\n";
            std::cout<<"Val Loss: "<<val_loss<<"\n";
            std::cout<<"Val Accuracy: "<<val_acc<<"\n";
            std::cout<<"Val F1 Score: "<<val_f1<<"\n";
            std::cout<<"Val ROC AUC: "<<val_auc<<"\n";

            total_accuracy+=val_acc;
            total_f1+=val_f1;
            total_auc+=val_auc;

            fold_num++;
        }

        std::cout<<"Средняя Accuracy: "<<total_accuracy/k<<"\n";
        std::cout<<"Средний F1 Score: "<<total_f1/k<<"\n";
        std::cout<<"Средний ROC AUC: "<<total_auc/k<<"\n";
    } catch(const std::exception &ex){
        Logger::error(std::string("Исключение: ")+ex.what());
    }

    Logger::close();
    return 0;
}