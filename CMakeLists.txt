cmake_minimum_required(VERSION 3.10)

project(CNN_MNIST_Project)

# Установка стандарта C++
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Поиск Eigen через пакет
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Добавление директорий с заголовочными файлами
include_directories(include ${EIGEN3_INCLUDE_DIR})

# Сборка исходных файлов
file(GLOB SRC_FILES 
    "src/*.cpp"
    "src/layers/*.cpp"
    "src/utils/*.cpp"
)

# Создание исполняемого файла
add_executable(cnn_mnist ${SRC_FILES})