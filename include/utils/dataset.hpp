#pragma once
#include "matrix.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

struct MNISTImage {
    Matrix<float> pixels; // 28x28
    int label;
};

class MNISTDataset {
public:
    static std::vector<MNISTImage> load_mnist(const std::string& image_path,const std::string& label_path){
        std::vector<MNISTImage> dataset;

        std::ifstream images_file(image_path,std::ios::binary);
        std::ifstream labels_file(label_path,std::ios::binary);

        if(!images_file.is_open())
            throw std::runtime_error("Не удалось открыть файл изображений: "+image_path);
        if(!labels_file.is_open())
            throw std::runtime_error("Не удалось открыть файл меток: "+label_path);

        int32_t magic=0;
        if(!read_int(images_file,magic)) throw std::runtime_error("Ошибка чтения magic num для изображений");
        if(magic!=2051) throw std::runtime_error("Неверный магический номер для изображений");
        int32_t num_images=0; if(!read_int(images_file,num_images)) throw std::runtime_error("Err");
        int32_t num_rows=0; if(!read_int(images_file,num_rows))throw std::runtime_error("Err");
        int32_t num_cols=0; if(!read_int(images_file,num_cols))throw std::runtime_error("Err");

        if(!read_int(labels_file,magic))throw std::runtime_error("Err reading label magic");
        if(magic!=2049) throw std::runtime_error("Неверный магический номер для меток");
        int32_t num_images_labels=0; if(!read_int(labels_file,num_images_labels))throw std::runtime_error("Err labels");
        if(num_images!=num_images_labels) throw std::runtime_error("Количество изображений и меток не совпадает");

        dataset.reserve(num_images);
        for(int i=0;i<num_images;++i){
            MNISTImage img;
            img.pixels=Matrix<float>(num_rows,num_cols,0);
            for(int r=0;r<num_rows;++r){
                for(int c=0;c<num_cols;++c){
                    unsigned char temp=0;
                    if(!images_file.read((char*)&temp,1)) throw std::runtime_error("Err чтения пикселя");
                    img.pixels(r,c)=temp/255.0f;
                }
            }
            unsigned char lbl=0;
            if(!labels_file.read((char*)&lbl,1)) throw std::runtime_error("Err reading label");
            img.label=(int)lbl;
            dataset.push_back(img);
        }

        images_file.close();
        labels_file.close();
        return dataset;
    }

private:
    static int32_t reverse_int(int32_t i){
        unsigned char c1,c2,c3,c4;
        c1=i&255;
        c2=(i>>8)&255;
        c3=(i>>16)&255;
        c4=(i>>24)&255;
        return ((int32_t)c1<<24)+((int32_t)c2<<16)+((int32_t)c3<<8)+c4;
    }

    static bool read_int(std::ifstream &f,int32_t &val){
        if(!f.read((char*)&val,4)) return false;
        val=reverse_int(val);
        return true;
    }
};