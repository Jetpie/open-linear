#include "high_level_function.hpp"
#include "logistic.hpp"
using namespace oplin;
using namespace std;
using namespace Eigen;
int main(int argc, char** argv)
{
    std::cout.precision(10);
    string filename = "res/sample";
    string output = "res/output";
    DatasetPtr dataset = read_dataset(filename,-1,270);
    const string model_name = "res/testmodel";
    ModelUniPtr model = read_model(model_name);
    shared_ptr<LinearBase> lr= make_shared<LogisticRegression>(std::move(model));

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

    predict_all(filename,output,lr," ",true);

}
