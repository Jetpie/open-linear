#include "command_utils.hpp"
#include "logistic.hpp"

int main(int argc, char** argv)
{
    string filename = "res/sample";
    size_t dimension = 13;

    DatasetPtr dataset = read_dataset(filename,dimension,270);
    ParamPtr param = make_shared<Parameter>();
    param->bias = 0.001;

    LogisticRegression lr;
    lr.train(dataset, param);

}
