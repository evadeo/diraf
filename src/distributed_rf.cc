#include "distributed_rf.hh"

DistributedRF::DistributedRF(int n_estimators, const std::string& criterion, int max_depth)
    : n_estimators_(n_estimators), criterion_(criterion), max_depth_(max_depth)
{
/*
    MPI_Init(nullptr, nullptr);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
*/
}

DistributedRF::~DistributedRF()
{
/*
    MPI_Finalize();
*/
}


int DistributedRF::fit()
{
    /*
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, features.size());

    // On récupère les features random pour cet arbre de décision
    std::set<int> features_index;
    for (int i = 0; i < n_estimators_; ++i)
        if (!features_index.insert(dis(gen)).second)
            --i;

    //TODO: Récupérer seulement les features avec les index choisis aléatoirement
    */
    return 0;
}

void DistributedRF::predict()
{

}
