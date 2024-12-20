#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <mutex>

class Logger {
private:
    static std::ofstream file_;
    static std::mutex mtx_;
public:
    static void init(const std::string& filename);
    static void log_metrics(size_t epoch,
                            float train_loss, float train_accuracy, float train_f1, float train_roc_auc,
                            float val_loss, float val_accuracy, float val_f1, float val_roc_auc);
    static void info(const std::string& msg);
    static void error(const std::string& msg);
    static void close();
};