#include "command_interface.hpp"
#include "logistic.hpp"

int main(int argc, char** argv)
{
    std::cout.precision(10);
    string filename = "res/sample";
    string output = "res/output";
    size_t dimension = 13;
    DatasetPtr dataset = read_dataset(filename,dimension,270);

    const string model_name = "res/sample.model";
    ModelPtr model = load_model(model_name);
    cout << "model" <<endl;
    double* W_ = model->W_;
    for(size_t i = 0; i < (model->n_classes-1) * model->dimension;++i)
    {
        cout << W_[i] << endl;
    }
    shared_ptr<LinearBase> lr= make_shared<LogisticRegression>(model);

    // cout << *(dataset->X) << endl;
    cout << dataset->n_samples <<endl;
    cout << dataset->dimension << endl;
    cout << dataset->n_classes << endl;

    cout << endl;
    //clock_t s;
    for(int k =0; k < (dataset->X)->outerSize(); ++k)
    {
        FeatureVector features;
        features.reserve(20);
        for(SparseMatrix<double>::InnerIterator it(*(dataset->X),k); it ;++it)
        {
            //cout <<"(" <<it.value() << "," << it.row()<<"," << it.col()<<")";
            features.push_back({(size_t)it.row(),it.value()});
        }
        //cout << endl;
        vector<double> p;
        //s = clock();
        double label = lr->predict_proba(features,p);
        //cout << "time test:" << float(clock() -s)/CLOCKS_PER_SEC << endl;
        //s = clock();
        label = lr->predict(features);
        //cout << "time test:" << float(clock() -s)/CLOCKS_PER_SEC << endl;
        cout << "prediction for column " << k+1 << " : "
             << label << " " << p[0] << "," << p[1] << endl;
    }
    predict_all(filename,output,lr);

}
