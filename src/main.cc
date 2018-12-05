#include "distributed_rf.hh"

#include <vector>

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    std::vector<int> features_a { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    std::vector<int> features_b { 1, 1, 1, 3, 4, 5, 8, 9, 4, 5, 2, 10, 20, 12, 7 };
    std::vector<int> features_c { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    std::vector<int> features_d { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    std::vector<int> features_e { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    std::vector<std::vector<int>> features;
    features.push_back(features_a);
    features.push_back(features_b);
    features.push_back(features_c);
    features.push_back(features_d);
    features.push_back(features_e);

    std::vector<int> labels { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };

    auto drf = DistributedRF(10, "gini", 10, 1);
    drf.fit(features, labels);
    //int res = drf.fit();
    //std::cout << "Got res: " << res << std::endl;
}
