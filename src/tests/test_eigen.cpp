#include "linear.hpp"
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <stdlib.h>
#include "Eigen/Dense"
#include <cmath>
using namespace oplin;
using namespace std;
using namespace Eigen;
typedef struct _dacommand {
    std::pair<unsigned int, uint64_t> _cmd;
    _dacommand(const char *cmd_str)
    {
        if (strncmp(cmd_str, "0x", 2) == 0 || strncmp(cmd_str, "0X", 2) == 0)
        {
            cmd_str += 2;
        }
        unsigned long long posoff = 0;
        const char *p;
        for (p = &cmd_str[strlen(cmd_str) - 1]; p >= cmd_str; p--)
        {
            if (*p != '0')
                break;
            else
                posoff++;
        }
        if (p >= cmd_str)
        {
            posoff *= 4;
            if (*p == '8')
                posoff += 3;
            else if (*p == '4')
                posoff += 2;
            else if (*p == '2')
                posoff += 1;
            else if (*p == '1')
                posoff += 0;
            else
                posoff = 0;
        }
        else
            posoff = 0;
        printf("posoff:%u\n", posoff);
        _cmd.first = posoff / 64;
        _cmd.second = (((unsigned long long)1UL << (posoff % 64)));
    }

    _dacommand(uint64_t cmd) { _cmd.first = 0; _cmd.second = cmd; }

    _dacommand(unsigned int offset, uint64_t cmd) { _cmd.first = offset; _cmd.second = cmd; }

    _dacommand(std::pair<unsigned int, uint64_t> sp) { _cmd = sp; }

    friend bool operator==(const _dacommand &ref1, const _dacommand &ref2);
    friend bool operator<(const _dacommand &ref1, const _dacommand &ref2);
    static uint64_t make_enum(char *instr){
        char *endptr = instr + strlen(instr) - 1;
        char *posptr = endptr;
        while (posptr >= instr && *posptr != 'X' && *posptr != 'x')
        {
            if (*posptr != '0')
                break;
            else
                posptr--;
        }
        uint64_t sum = 0;
        if (posptr >= instr && *posptr != 'X' && *posptr != 'x')
        {
            sum = (endptr - posptr) * 4;
            if (*posptr == '8')
               sum += 3;
            else if (*posptr == '4')
               sum += 2;
            else if (*posptr == '2')
               sum += 1;
            else if (*posptr == '1')
               sum += 0;
            else
               sum = 0;
        }
        return sum;
    }
} DACOMMAND;
int main()
{
    // DACOMMAND x1("0x0080");
    // printf("0x000\t%u\t%lu\n", x1._cmd.first, x1._cmd.second);
    // DACOMMAND x2("0x010");
    // printf("0x001\t%u\t%lu\n", x2._cmd.first, x2._cmd.second);
    // DACOMMAND x3("0x4000000000000000000");
    // printf("0x002\t%u\t%lu\n", x3._cmd.first, x3._cmd.second);

    int dx = 750;
    int x = 3;
    float scaler = 0.5;

    int a = (pow(dx, 3) * scaler / 10000000.0);
    printf("a: %d\n", a);


    // DACOMMAND x3("0x005");
    // printf("0x002\t%u\t%lu\n", x3._cmd.first, x3._cmd.second);
    // MatrixXf m(3,3);
    // m << 1,2,3,4,5,6,7,8,9;
    // m /= 1.5;
    // cout << m << endl;
    // SpColVector spv(1000);
    // SpRowMatrix spm(1000,1000);
    // SpColMatrix spm_col(1000,1000);
    // RowVector den_row_vec(1000);
    // ColVector den_col_vec(1000);
    // for (size_t i = 0; i < 1000 ; ++i)
    // {
    //     den_row_vec(i) = i*i;
    //     den_col_vec(i) = i*i;
    // }

    // clock_t start = clock();
    // for(size_t i = 0; i < 15 ;++i)
    // {
    //     spv.coeffRef(i) = 1.5;
    //     spv.coeffRef(i*i) = 2.5;
    // }

    // SpColVector copy_spv =
    // spv.coeffRef(10) = 1.5;
    // spv.coeffRef(999) = 1.5;
    // spv.coeffRef(100) = 2.5;
    // cout << copy_spv << endl;
    // cout << spv << endl;
    // cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    // cout << spv.squaredNorm() / 2.0 << endl;
    // for(size_t i = 0; i < 15 ;++i)
    // {
    //     spm.insert(i,i*i) = 1.5;
    //     spm.insert(i+5,i*i) = 1.5;
    //     spm_col.insert(i,i*i) = 1.5;
    //     spm_col.insert(i+5,i*i) = 1.5;
    // }
    // SpColMatrix spm1(10,10);
    // spm1.insert(9,9) = 101.5;
    // spm1.insert(6,9) = 105.5;
    // spm1.insert(8,8) = 1000.5;
    // spm1.makeCompressed();
    // cout << spm1 << endl;
    // int a[] = {3,2,1,5,6,9,8,7,4,0};
    // PermutationMatrix<Dynamic,Dynamic> perm_matrix(10);
    // start = clock();
    // for(int i =0 ;i < 10 ;i++)
    // {
    //     perm_matrix.indices()[i] = a[i];
    // }
    // // perm_matrix.setIdentity();
    // // perm_matrix.indices()[8] = 9;
    // // perm_matrix.indices()[9] = 8;
    // cout << perm_matrix.indices() << endl;

    // spm1=  (spm1* perm_matrix).eval();
    // cout << spm1 << endl;
    // cout << "time permutation:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    // cout << "swap test : " << spm1.coeffRef(9,8) << "," <<
    //     spm1.coeffRef(8,9) << endl;
    // // for (SparseVector<double>::InnerIterator it(spv); it; ++it)
    // // {
    // //     cout << it.value() << ","
    // //          << it.index() << endl;
    // // }
    // cout << spv << endl;
    // cout << spv.rows() << endl;
    // spm.makeCompressed();
    // cout << den_row_vec.cols() << endl;
    // // den_row_vec.noalias() =  den_row_vec.transpose();
    // // cout << drvt.cols() << endl;
    // start = clock();

    // cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;

    // start = clock();
    // SpRowVector res = spv.transpose() * spm;
    // cout << res.cols() << "," << res.rows()  << endl;
    // cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    // start = clock();
    // res =  spv.transpose() * spm;
    // cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    // start = clock();
    //  res =  spv.transpose() * spm;
    // cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;

    // cout << spm_col.rows() << "," << spm_col.cols() << endl;
    // cout << den_col_vec.rows() << "," << den_col_vec.cols() << endl;
    // start = clock();
    // double res1 =   den_col_vec.transpose() * den_col_vec;
    // // cout << res1.rows() << "," << res1.cols() << endl;
    // cout << res1 << endl;
    // cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    // start = clock();
    // RowVector res2 =  spm * den_row_vec.transpose() ;
    // cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    // cout << res2  <<endl;
    // cout << res2 * 0.0001 <<endl;

    // // Sparse by Dense Test
    // cout << ">>>>>>>>>>Sparse Matrix By Dense Vector<<<<<<<<<<"<<endl;
    // SpColMatrix l_m(10,10);
    // for(size_t i = 0;i < 10;++i)
    //     for(size_t j=0;j < 10; ++j)
    //         l_m.coeffRef(i,j) = i*j;
    // ColVector r_v = ColVector::Ones(10,1);
    // cout << "l_m" << endl;
    // cout << l_m << endl;
    // cout << "r_v" << endl;
    // cout << r_v << endl;
    // ColVector result_mv = l_m * r_v;
    // cout << "result" << endl;
    // cout << result_mv << endl;

    // // // Random Test
    // // cout << ">>>>>>>>>>Random Test<<<<<<<<<<"<<endl;

    // // for(int i = 0; i < 5; i++) {
    // //     srand((unsigned int) time(0));
    // //     std::this_thread::sleep_for (std::chrono::seconds(1));
    // //     ColMatrix A = ColMatrix::Random(3, 3) / 2;
    // //     cout << A <<endl;
    // // }

    // std::string sss = "123:1.15";
    // const char* item = sss.c_str();
    // int i;
    // float v_ij;
    // // sscanf(item.c_str(),"%d:%f",&i,&v_ij);
    // char *p_end;
    // i = (int) strtol(item,&p_end,10);
    // v_ij = strtod(p_end,NULL);
    // // i = atoi(strtok(item,":") );
    // // v_ij = atof(strtok(NULL,":" ));
    // cout <<i <<"," << v_ij << endl;
    // // #pragma omp parallel num_threads(5)
    // // {
    // //     #pragma omp for
    // //     for(int n=0;n<10;++n) cout << n << endl;
    // // }
}
