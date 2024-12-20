#include "../../include/utils/logger.hpp"

std::ofstream Logger::file_;
std::mutex Logger::mtx_;

void Logger::init(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mtx_);
    file_.open(filename, std::ios::out);
    if(!file_.is_open()){
        std::cerr<<"Не удалось открыть файл логирования: "<<filename<<"\n";
    } else {
        file_<<"Epoch,Train_Loss,Train_Accuracy,Train_F1,Train_ROC_AUC,Val_Loss,Val_Accuracy,Val_F1,Val_ROC_AUC\n";
    }
}

void Logger::log_metrics(size_t epoch,
                         float train_loss, float train_accuracy, float train_f1, float train_roc_auc,
                         float val_loss, float val_accuracy, float val_f1, float val_roc_auc) {
    std::lock_guard<std::mutex> lock(mtx_);
    if(file_.is_open()){
        file_<<epoch<<","
             <<train_loss<<","<<train_accuracy<<","<<train_f1<<","<<train_roc_auc<<","
             <<val_loss<<","<<val_accuracy<<","<<val_f1<<","<<val_roc_auc<<"\n";
    }
}

void Logger::info(const std::string& msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    std::cout<<"[INFO] "<<msg<<"\n";
    if(file_.is_open()){
        file_<<"[INFO] "<<msg<<"\n";
    }
}

void Logger::error(const std::string& msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    std::cerr<<"[ERROR] "<<msg<<"\n";
    if(file_.is_open()){
        file_<<"[ERROR] "<<msg<<"\n";
    }
}

void Logger::close() {
    std::lock_guard<std::mutex> lock(mtx_);
    if(file_.is_open()) file_.close();
}