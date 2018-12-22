#pragma once

#include <vector>

#include "error_struct.hh"

float gini_error(std::vector<int> count_label_vector);
Error total_gini_error(const std::vector<int>& features, const std::vector<int>& labels,
        int split_value, size_t split_index);
