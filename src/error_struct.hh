#pragma once

/**
 * \struct Error
 *
 * \brief Error struct to regroup all information
 *
 * This struct was made in order to regroup multiple information 
 * and simplify function return values.
 *
 */

#include <cstddef>

struct Error
{
    ///Integer representing the split value
    int split_value_;
    ///Unsigned integer representing the index on which the split should be done
    size_t split_index_;
    ///Float value representing the error of the left split
    float err_left_;
    ///Float value representing the error of the right split
    float err_right_;
    ///Float value representing the total error for this split
    float err_total_;

    /**
     * \brief Constructor for the error class
     */
    Error(int split_value = 0, size_t split_index = 0,
          float err_left = 1, float err_right = 1, float err_total = 1)
        : split_value_(split_value)
        , split_index_(split_index)
        , err_left_(err_left)
        , err_right_(err_right)
        , err_total_(err_total)
    {}

    /**
     * \brief assignement operator
     */
    Error& operator =(const Error& e)
    {
        split_value_ = e.split_value_;
        split_index_ = e.split_index_;
        err_left_ = e.err_left_;
        err_right_ = e.err_right_;
        err_total_ = e.err_total_;
        return *this;
    }

    /**
     * \brief Lower than comparaison operator
     */
    bool operator <(const Error& e) const
    {
        return err_total_ < e.err_total_;
    }
};
