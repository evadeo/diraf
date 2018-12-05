#pragma once

#include <functional>
#include <vector>

std::function<float(std::vector<int>)> get_error_function(const std::string& criterion);
int get_number_of_elems(std::vector<int> count_label_vector);
