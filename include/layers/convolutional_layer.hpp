#pragma once
#include "layer.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

template<typename T>
class ConvolutionalLayer : public Layer<T> {
private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;

    std::vector<Matrix<T>> kernels_; // размер out_channels_ * in_channels_
    std::vector<T> biases_;

    std::vector<std::vector<Matrix<T>>> input_cache_;

public:
    ConvolutionalLayer(int in_channels,int out_channels,int kernel_size,int stride=1,int padding=0)
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
          stride_(stride), padding_(padding) {
        initialize_kernels();
    }

    std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& input) override {
        if((int)input.size()!=in_channels_){
            throw std::runtime_error("ConvolutionalLayer: неверное число входных каналов.");
        }
        input_cache_.push_back(input);

        int input_height=(int)input[0].rows();
        int input_width=(int)input[0].cols();

        int output_height=(input_height - kernel_size_ + 2*padding_)/stride_+1;
        int output_width=(input_width - kernel_size_ + 2*padding_)/stride_+1;

        std::vector<Matrix<T>> output_channels;
        output_channels.reserve(out_channels_);
        for(int out_c=0;out_c<out_channels_;++out_c){
            Matrix<T> out_ch(output_height,output_width,0);
            for(int in_c=0;in_c<in_channels_;++in_c){
                Matrix<T> convolved=convolve(input[in_c],kernels_[out_c*in_channels_+in_c],stride_,padding_);
                // сложение convolved в out_ch
                for(int i=0;i<output_height;++i){
                    for(int j=0;j<output_width;++j){
                        out_ch(i,j)+=convolved(i,j);
                    }
                }
            }
            // Добавляем смещение
            for(int i=0;i<output_height;++i){
                for(int j=0;j<output_width;++j){
                    out_ch(i,j)+=biases_[out_c];
                }
            }
            output_channels.push_back(out_ch);
        }

        return output_channels;
    }

    std::vector<Matrix<T>> backward(const std::vector<Matrix<T>>& dLoss, T learning_rate, T lambda=0.0) override {
        if((int)dLoss.size()!=out_channels_){
            throw std::runtime_error("ConvolutionalLayer backward: неверное число выходных каналов.");
        }

        const std::vector<Matrix<T>>& input=input_cache_.back();
        input_cache_.pop_back();

        int input_height=(int)input[0].rows();
        int input_width=(int)input[0].cols();

        std::vector<Matrix<T>> grad_input(in_channels_, Matrix<T>(input_height,input_width,0));
        std::vector<Matrix<T>> grad_kernels(out_channels_*in_channels_, Matrix<T>(kernel_size_,kernel_size_,0));
        std::vector<T> grad_biases(out_channels_,0);

        for(int out_c=0;out_c<out_channels_;++out_c){
            for(int in_c=0;in_c<in_channels_;++in_c){
                Matrix<T> gk=compute_grad_kernel(input[in_c],dLoss[out_c],stride_,padding_);
                for(int rr=0;rr<kernel_size_;++rr){
                    for(int cc=0;cc<kernel_size_;++cc){
                        grad_kernels[out_c*in_channels_+in_c](rr,cc)=gk(rr,cc);
                    }
                }

                Matrix<T> gi=compute_grad_input(dLoss[out_c],kernels_[out_c*in_channels_+in_c],stride_,padding_);
                for(int rr=0;rr<input_height;++rr){
                    for(int cc=0;cc<input_width;++cc){
                        grad_input[in_c](rr,cc)+=gi(rr,cc);
                    }
                }
            }
            // grad по смещениям
            for(int i=0;i<(int)dLoss[out_c].rows();++i){
                for(int j=0;j<(int)dLoss[out_c].cols();++j){
                    grad_biases[out_c]+=dLoss[out_c](i,j);
                }
            }
        }

        // Обновление параметров
        for(int i=0;i<out_channels_*in_channels_;++i){
            if(lambda>0){
                for(int rr=0;rr<kernel_size_;++rr){
                    for(int cc=0;cc<kernel_size_;++cc){
                        grad_kernels[i](rr,cc)+=lambda*kernels_[i](rr,cc);
                    }
                }
            }

            for(int rr=0;rr<kernel_size_;++rr){
                for(int cc=0;cc<kernel_size_;++cc){
                    kernels_[i](rr,cc)-=learning_rate*grad_kernels[i](rr,cc);
                }
            }
        }

        for(int out_c=0;out_c<out_channels_;++out_c){
            if(lambda>0) grad_biases[out_c]+=lambda*biases_[out_c];
            biases_[out_c]-=learning_rate*grad_biases[out_c];
        }

        return grad_input;
    }

private:
    void initialize_kernels(){
        std::mt19937 gen(std::random_device{}());
        T stddev=std::sqrt((T)2.0/(T)(in_channels_*kernel_size_*kernel_size_));
        std::normal_distribution<T> dist(0,stddev);

        for(int i=0;i<out_channels_*in_channels_;++i){
            Matrix<T> k(kernel_size_,kernel_size_,0);
            for(int m=0;m<kernel_size_;++m){
                for(int n=0;n<kernel_size_;++n){
                    k(m,n)=dist(gen);
                }
            }
            kernels_.push_back(k);
        }

        for(int i=0;i<out_channels_;++i){
            biases_.push_back((T)0);
        }
    }

    Matrix<T> convolve(const Matrix<T>& input,const Matrix<T>& kernel,int stride,int padding){
        int padded_height=(int)input.rows()+2*padding;
        int padded_width=(int)input.cols()+2*padding;
        Matrix<T> padded(padded_height,padded_width,0);
        for(size_t i=0;i<input.rows();++i){
            for(size_t j=0;j<input.cols();++j){
                padded(i+padding,j+padding)=input(i,j);
            }
        }

        int output_height=(padded_height-kernel_size_)/stride+1;
        int output_width=(padded_width-kernel_size_)/stride+1;
        Matrix<T> out(output_height,output_width,0);
        for(int i=0;i<output_height;++i){
            for(int j=0;j<output_width;++j){
                T sum=0;
                for(int m=0;m<kernel_size_;++m){
                    for(int n=0;n<kernel_size_;++n){
                        sum+=padded(i*stride+m,j*stride+n)*kernel(m,n);
                    }
                }
                out(i,j)=sum;
            }
        }

        return out;
    }

    Matrix<T> compute_grad_kernel(const Matrix<T>& input,const Matrix<T>& dLoss,int stride,int padding){
        int padded_height=(int)input.rows()+2*padding;
        int padded_width=(int)input.cols()+2*padding;
        Matrix<T> padded(padded_height,padded_width,0);
        for(size_t i=0;i<input.rows();++i){
            for(size_t j=0;j<input.cols();++j){
                padded(i+padding,j+padding)=input(i,j);
            }
        }

        Matrix<T> grad_k(kernel_size_,kernel_size_,0);
        for(int m=0;m<kernel_size_;++m){
            for(int n=0;n<kernel_size_;++n){
                T sum=0;
                for(int i=0;i<(int)dLoss.rows();++i){
                    for(int j=0;j<(int)dLoss.cols();++j){
                        sum+=padded(i*stride+m,j*stride+n)*dLoss(i,j);
                    }
                }
                grad_k(m,n)=sum;
            }
        }
        return grad_k;
    }

    Matrix<T> compute_grad_input(const Matrix<T>& dLoss,const Matrix<T>& kernel,int stride,int padding){
        Matrix<T> flipped=flip_matrix(kernel);

        int grad_input_height=(dLoss.rows()-1)*stride - 2*padding + kernel_size_;
        int grad_input_width=(dLoss.cols()-1)*stride - 2*padding + kernel_size_;

        Matrix<T> grad_in(grad_input_height,grad_input_width,0);
        for(int i=0;i<(int)dLoss.rows();++i){
            for(int j=0;j<(int)dLoss.cols();++j){
                for(int m=0;m<kernel_size_;++m){
                    for(int n=0;n<kernel_size_;++n){
                        int x=i*stride+m-padding;
                        int y=j*stride+n-padding;
                        if(x>=0&&x<grad_input_height&&y>=0&&y<grad_input_width){
                            grad_in(x,y)+=dLoss(i,j)*flipped(m,n);
                        }
                    }
                }
            }
        }

        return grad_in;
    }

    Matrix<T> flip_matrix(const Matrix<T>& mat){
        Matrix<T> flipped(mat.rows(),mat.cols(),0);
        for(int i=0;i<(int)mat.rows();++i){
            for(int j=0;j<(int)mat.cols();++j){
                flipped(i,j)=mat(mat.rows()-1-i,mat.cols()-1-j);
            }
        }
        return flipped;
    }
};