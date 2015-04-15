#include "command_utils.hpp"


int main(int argc, char** argv)
{
    std::cout.precision(10);
    string filename = "res/sample";
    size_t dimension = 13;

    DatasetPtr dataset = read_dataset(filename,dimension,270);


    // cout << *(dataset->X) << endl;
    cout << dataset->n_samples <<endl;
    cout << dataset->dimension << endl;
    cout << dataset->n_classes << endl;
    for(size_t i = 0;i<dataset->n_classes;++i)
    {
        cout << dataset->labels[i] << ",";

    }
    cout << endl;
    for(int k =0; k < (dataset->X)->outerSize(); ++k)
    {
        for(SparseMatrix<double>::InnerIterator it(*(dataset->X),k); it ;++it)
        {
            cout <<"(" <<it.value() << "," << it.row()<<"," << it.col()<<")";
        }
        cout << endl;
    }
    cout << endl;
    for(size_t i = 0; i<dataset->n_samples;++i)
    {
        cout << dataset->y[i] <<",";
    }
    cout << endl;

    const string model_name = "res/sample.model";
    ModelPtr model = load_model(model_name);
    cout << "model" <<endl;
    cout << (*model->W) <<endl;

}
