#pragma once
#include <stdexcept>
#include <string>

class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& msg)
        : std::runtime_error(msg) {}
};