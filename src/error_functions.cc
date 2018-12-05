#include "error_functions.hh"
#include "utils.hh"
#include <cmath>
//#include <execution>

float gini_error(std::vector<int> count_label_vector)
{

    /*
     * Gini err is defined as follows:
     * gini_err = 1 - sum((number_of_element_in_class_i / total_number_of_elem)²)
     */

    //Reduce est plus rapide mais faudrait voir avec le flag c++1z
    //int elem_count = std::reduce(std::execution::par, count_label_vector.begin(),
    //                             count_label_vector.end());
    int elem_count = get_number_of_elems(count_label_vector);
    if (elem_count == 0)
        return 1;

    float gini_err = 1;
    for (size_t i = 0; i < count_label_vector.size(); ++i)
        gini_err -= std::pow((count_label_vector[i] / elem_count), 2);

    return gini_err;
}
