#pragma once
#include <vector>
#include <utility>
#include <stdexcept>
#include <algorithm>

class CrossValidator {
public:
    template<typename T>
    static std::vector<std::pair<std::vector<T>,std::vector<T>>> k_fold_split(const std::vector<T>& data,size_t k){
        if(k==0) throw std::runtime_error("k must be >0");
        size_t n=data.size();
        if(k>n) k=n;
        std::vector<std::pair<std::vector<T>,std::vector<T>>> folds;
        size_t fold_size=n/k;
        size_t start=0;
        for(size_t i=0;i<k;++i){
            size_t end=(i==k-1)?n:start+fold_size;
            std::vector<T> val(data.begin()+start,data.begin()+end);
            std::vector<T> train;
            train.reserve(n-(end-start));
            if(start>0) train.insert(train.end(),data.begin(),data.begin()+start);
            if(end<n) train.insert(train.end(),data.begin()+end,data.end());
            folds.push_back({train,val});
            start=end;
        }
        return folds;
    }
};