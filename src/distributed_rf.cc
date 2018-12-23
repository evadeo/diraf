#include <algorithm>
#include <utility>
#include "distributed_rf.hh"

DistributedRF::DistributedRF(int n_estimators, const std::string& criterion, int max_depth,
        int max_features, bool distributed)
    : criterion_(criterion)
    , max_depth_(max_depth)
    , max_features_(max_features)
    , distributed_(distributed)
{
    ///If the program is not distributed we only need to assign a few variables.
    if (!distributed)
    {
        n_estimators_ = n_estimators;
        trees_ = std::vector<DecisionTree>();
        predictions_ = std::vector<int>();
    }
    else
    {
        ///Start the distributed environment
        MPI_Init(nullptr, nullptr);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);

        int criterion_size;
        char* criterion_str;
        if (rank_ == 0)
        {
            ///Compute the number of trees for each instance
            n_estimators_ = n_estimators / size_;
            criterion_size = criterion_.size();
        }

        ///Broadcasting all the the data to all the instance
        MPI_Bcast(&n_estimators_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&max_depth_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&max_features_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&criterion_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        ///Broadcast criterion string
        criterion_str = (char*)calloc(1, criterion_size + 1);
        if (rank_ == 0)
            std::copy(criterion.data(), criterion.data() + criterion_size, criterion_str);
        MPI_Bcast(criterion_str, criterion_size, MPI_CHAR, 0, MPI_COMM_WORLD);

        ///Assign the criterion string to the attribute for each instance
        criterion_ = std::string(criterion_str);
        free(criterion_str);

        ///Instanciate the vectors
        trees_ = std::vector<DecisionTree>();
        predictions_ = std::vector<int>();

        if (rank_ != 0)
        {
            ///If we are not the master node, go to the looper and wait for next instruction.
            looper();
            ///Once it is finished, the master node will have finished the execution, so we call mpi_finalize on each other instances
            MPI_Finalize();
        }
    }
}

void DistributedRF::looper()
{
    bool cont = true;
    ///While we do not receive the EXIT command
    while (cont)
    {
        enum CallMeMaybe cmm;
        ///Receive the next instruction from master node
        MPI_Recv(&cmm, sizeof(enum CallMeMaybe), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        auto zero = std::vector<int>(0);
        auto zerozero = std::vector<std::vector<int>>(0);
        ///Call the right function according to the enum
        switch (cmm)
        {
            case FIT:
                distributed_fit(zerozero, zero);
                break;

            case PREDICT:
                distributed_predict(zerozero);
                break;

            case EXIT:
                cont = false;
                break;
        }
    }
}


DistributedRF::~DistributedRF()
{
    ///If we weren't launch in distributed mode, we do not want to call mpi_finalize
    if (distributed_)
    {
        ///The destructor will first be called by the master node because all the other nodes are in the looper.
        if (rank_ == 0)
        {
            ///Send to all nodes the EXIT command.
            enum CallMeMaybe cmm = CallMeMaybe::EXIT;
            for (int i = 1; i < size_; ++i)
            {
                MPI_Send(&cmm, sizeof(enum CallMeMaybe), MPI_BYTE, i, 0, MPI_COMM_WORLD);
            }
            ///Call finalize for the master node.
            MPI_Finalize();

        }
    }
}


static std::vector<std::vector<int>> get_random_features(
        const std::vector<std::vector<int>>& features,
        std::vector<int>& real_indexes,
        int max_features)
{
    int m_features = max_features == - 1 ? features[0].size(): max_features;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, features[0].size() - 1);

    std::set<int> features_index;
    for (int i = 0; i < m_features; ++i)
        if (!features_index.insert(dis(gen)).second)
            --i;

    std::copy(features_index.begin(), features_index.end(), std::back_inserter(real_indexes));

    std::vector<std::vector<int>> random_features(features.size());
    for (size_t i = 0; i < features.size(); ++i)
        for (auto f_index : features_index)
            random_features[i].push_back(features[i][f_index]);
    return random_features;
}


void DistributedRF::distributed_fit(const std::vector<std::vector<int>>& features_root,
        const std::vector<int>& labels_root)
{
    ///Check if we did not call mpi_finalized first
    int flag;
    MPI_Finalized(&flag);
    if (flag)
        return;

    int nbFeats, nbValue, nbLabels;
    std::vector<int> labels;
    std::vector<std::vector<int>> features;
    if (rank_ == 0)
    {
        ///Initialize the values to broadcast
        nbFeats = features_root.size();
        nbValue = features_root[0].size();
        nbLabels = labels_root.size();
        labels = std::vector<int>(labels_root);
        features = std::vector<std::vector<int>>(features_root);

        ///Send to all instances to go to the distributed FIT function
        enum CallMeMaybe cmm = CallMeMaybe::FIT;
        for (int i = 1; i < size_; ++i)
        {
            MPI_Send(&cmm, sizeof(enum CallMeMaybe), MPI_BYTE, i, 0, MPI_COMM_WORLD);
        }
    }
    ///Broadcast all the data
    MPI_Bcast(&nbFeats, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nbValue, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nbLabels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank_ != 0)
    {
        ///Build the vectors to have enough space when receiving the broadcasted data
        features = std::vector<std::vector<int>>(nbFeats, std::vector<int>(nbValue));
        labels = std::vector<int>(nbLabels);
    }

    ///Broadcast all the elemnts features to all the instances.
    for (int i = 0; i < nbFeats; ++i)
        MPI_Bcast(features[i].data(), nbValue, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(labels.data(), nbLabels, MPI_INT, 0, MPI_COMM_WORLD);

    ///Executing the fit method for all instances that will build the `trees_` attributed for each instance.
    this->fit(features, labels);

    ///The master node will return to the main program and all the other instances will go back to the looper function.
}


void DistributedRF::fit(const std::vector<std::vector<int>>& features,
        const std::vector<int>& labels)
{
    for (int i = 0; i < n_estimators_; ++i)
    {
        // On récupère les features random pour cet arbre de décision
        std::vector<int> real_indexes;
        auto random_features = get_random_features(features, real_indexes, max_features_);
        auto err_function = get_error_function(criterion_);
        trees_.emplace_back(random_features, labels, real_indexes, err_function);
    }
}

static int choose_prediction(const std::vector<int>& predictions)
{
    std::map<int, int> counts;

    for (int pred : predictions)
        ++counts[pred];

    const auto max = std::max_element(counts.begin(), counts.end(),
            [](const auto& p1, const auto& p2) { return p1.second < p2.second; });

    return max->first;
}

int DistributedRF::predict_label(const std::vector<int>& elem)
{
    std::vector<int> predictions;
    for (const auto& node : trees_)
        predictions.push_back(node.predict(elem));

    int pred = choose_prediction(predictions);

    return pred;
}

std::vector<int> DistributedRF::predict(const std::vector<std::vector<int>>& features)
{
    std::vector<int> predictions;

    for (size_t i = 0; i < features.size(); ++i)
    {
        int pred = this->predict_label(features[i]);
        predictions.push_back(pred);
    }
    return predictions;
}

void DistributedRF::distributed_predict(const std::vector<std::vector<int>>& features_root)
{
    ///Check if we did not call mpi_finalized first
    int flag;
    MPI_Finalized(&flag);
    if (flag)
        return;
    int nbFeats, nbValue;
    std::vector<std::vector<int>> features;
    if (rank_ == 0)
    {
        ///Initialize the values to broadcast
        nbFeats = features_root.size();
        nbValue = features_root[0].size();
        features = std::vector<std::vector<int>>(features_root);
        ///Send to all instances to go to the distributed FIT function
        enum CallMeMaybe cmm = CallMeMaybe::PREDICT;
        for (int i = 1; i < size_; ++i)
        {
            MPI_Send(&cmm, sizeof(enum CallMeMaybe), MPI_BYTE, i, 0, MPI_COMM_WORLD);
        }
    }

    ///Broadcast the values
    MPI_Bcast(&nbFeats, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nbValue, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank_ != 0)
        features = std::vector<std::vector<int>>(nbFeats, std::vector<int>(nbValue));
    for (int i = 0; i < nbFeats; ++i)
        MPI_Bcast(features[i].data(), nbValue, MPI_INT, 0, MPI_COMM_WORLD);

    ///Get the predictions for the elements on all instances
    std::vector<int> predictions = this->predict(features);

    ///Retrieving all information on master node
    if (rank_ == 0)
    {
        ///Vector that will contain all the predictions of each instance
        std::vector<std::vector<int>> all_preds(size_, std::vector<int>(predictions.size()));
        ///The first vector will be the predictions made by the master node
        all_preds[0] = predictions;
        ///Receive all the predictions from all instances
        for (int i = 1; i < size_; ++i)
            MPI_Recv(all_preds[i].data(), predictions.size(), MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        ///Select the most frequent output class for each element
        for (size_t i = 0; i < all_preds[0].size(); ++i)
        {
            std::vector<int> pred_elem;
            for (size_t j = 0; j < all_preds.size(); ++j)
                pred_elem.push_back(all_preds[j][i]);
            predictions_.push_back(choose_prediction(pred_elem));
        }
    }
    else
    {
        ///If we are not the master node, we send our predictions to it.
        MPI_Send(predictions.data(), predictions.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

std::vector<int> DistributedRF::get_predictions() const
{
    ///Check if we did not call mpi_finalized first
    int flag;
    MPI_Finalized(&flag);
    if (flag)
        return std::vector<int>();
    ///Returns the previously built predictions
    return predictions_;
}
