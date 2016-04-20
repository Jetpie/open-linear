#include "high_level_function.hpp"
#include "logistic.hpp"
#include <sstream>
#include <cstdlib>
using namespace oplin;
using namespace std;
using namespace Eigen;

int main(int argc, char** argv)
{
    std::string x("0x123");
    std::istringstream ss_x(x);
    char* p_end;
    int num;
    num = strtol(x.c_str(), &p_end, 10);
    if(*p_end==0)
    {
        cout << "numX:" << num<<endl;

    }
    else
    {
        cout << "bad casting x" << endl;
    }


    ss_x >> num;
    if(!ss_x.eof())
    {
        cout <<"bad casting x" << endl;
    }
    else
    {
        cout << "numX:" << num<<endl;
    }
    std::string y("2.5");
    char* y_end;
    float num_y;
    num_y = strtod(y.c_str(), &y_end);
    if(*y_end!=0)
    {
        cout <<"bad casting y" << endl;
    }
    else
    {
        cout << "numY:" << num_y<<endl;
    }


    return 0;
    // std::cout.precision(10);
    // string filename = "res/sample";
    // string output = "res/output";
    // DatasetPtr dataset = read_dataset(filename,-1,270);
    // const string model_name = "res/testmodel";
    // ModelUniPtr model = read_model(model_name);
    // shared_ptr<LinearBase> lr= make_shared<LogisticRegression>(std::move(model));

    // //clock_t s;
    // for(int k =0; k < (dataset->X)->outerSize(); ++k)
    // {
    //     FeatureVector features;
    //     features.reserve(20);
    //     for(SparseMatrix<double>::InnerIterator it(*(dataset->X),k); it ;++it)
    //     {
    //         //cout <<"(" <<it.value() << "," << it.row()<<"," << it.col()<<")";
    //         features.push_back({(size_t)it.row(),it.value()});
    //     }
    //     //cout << endl;
    //     vector<double> p;
    //     //s = clock();
    //     double label = lr->predict_proba(features,p);
    //     //cout << "time test:" << float(clock() -s)/CLOCKS_PER_SEC << endl;
    //     //s = clock();
    //     label = lr->predict(features);
    //     //cout << "time test:" << float(clock() -s)/CLOCKS_PER_SEC << endl;
    //     cout << "prediction for column " << k+1 << " : "
    //          << label << " " << p[0] << "," << p[1] << endl;
    // }

    // predict_all(filename,output,lr," ",true);


}
