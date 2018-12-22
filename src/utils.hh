#pragma once

#include <functional>
#include <vector>

#include "error_struct.hh"

typedef std::function<Error(const std::vector<int>&, const std::vector<int>&, int, size_t)> err_func;

err_func get_error_function(const std::string& criterion);
int get_number_of_elems(std::vector<int> count_label_vector);
std::vector<int> get_count_by_label(std::vector<int> feature_values,
                                     std::vector<int> labels,
                                     int split_value,
                                     const std::function<bool(int, int)>& comp);
int get_label_for_left_split(const std::vector<int>& features,
                             const std::vector<int>& labels, int split_value);
int get_label_for_right_split(const std::vector<int>& features,
                             const std::vector<int>& labels, int split_value);
