#pragma once

#include <cstddef>

struct Error
{
    int split_value_;
    size_t split_index_;
    float err_left_;
    float err_right_;
    float err_total_;

    Error(int split_value = 0, size_t split_index = 0,
          float err_left = 1, float err_right = 1, float err_total = 1)
        : split_value_(split_value)
        , split_index_(split_index)
        , err_left_(err_left)
        , err_right_(err_right)
        , err_total_(err_total)
    {}

    Error& operator =(const Error& e)
    {
        split_value_ = e.split_value_;
        split_index_ = e.split_index_;
        err_left_ = e.err_left_;
        err_right_ = e.err_right_;
        err_total_ = e.err_total_;
        return *this;
    }

    bool operator <(const Error& e) const
    {
        return err_total_ < e.err_total_;
    }
};
