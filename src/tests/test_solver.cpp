#include "solver.hpp"


int main()
{
    CRS_ColVector spv(1000);
    CRS_RowMatrix spm(1000,1000);
    CRS_ColMatrix spm_col(1000,1000);
    RowVector den_row_vec(1000);
    ColVector den_col_vec(1000);
    for (size_t i = 0; i < 1000 ; ++i)
    {
        den_row_vec(i) = i*i;
        den_col_vec(i) = i*i;
    }

    clock_t start = clock();
    for(size_t i = 0; i < 15 ;++i)
    {
        spv.coeffRef(i) = 1.5;
        spv.coeffRef(i*i) = 2.5;
    }

    CRS_RowVector copy_spv = spv;
    spv.coeffRef(10) = 1.5;
    spv.coeffRef(999) = 1.5;
    spv.coeffRef(100) = 2.5;
    cout << copy_spv << endl;
    cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    cout << spv.squaredNorm() / 2.0 << endl;
    for(size_t i = 0; i < 15 ;++i)
    {
        spm.insert(i,i*i) = 1.5;
        spm.insert(i+5,i*i) = 1.5;
        spm_col.insert(i,i*i) = 1.5;
        spm_col.insert(i+5,i*i) = 1.5;
    }
    // for (SparseVector<double>::InnerIterator it(spv); it; ++it)
    // {
    //     cout << it.value() << ","
    //          << it.index() << endl;
    // }
    cout << spv << endl;
    cout << spv.rows() << endl;
    spm.makeCompressed();
    cout << den_row_vec.cols() << endl;
    // den_row_vec.noalias() =  den_row_vec.transpose();
    // cout << drvt.cols() << endl;
    start = clock();

    cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;

    start = clock();
    CRS_RowVector res = spv.transpose() * spm;
    cout << res.cols() << "," << res.rows()  << endl;
    cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    start = clock();
    res =  spv.transpose() * spm;
    cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    start = clock();
     res =  spv.transpose() * spm;
    cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    start = clock();
     res =  spv.transpose() * spm;
    cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;

    cout << spm_col.rows() << "," << spm_col.cols() << endl;
    cout << den_col_vec.rows() << "," << den_col_vec.cols() << endl;
    start = clock();
    double res1 =   den_col_vec.transpose() * den_col_vec;
    // cout << res1.rows() << "," << res1.cols() << endl;
    // cout << res1 << endl;
    cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    start = clock();
    RowVector res2 =  spm * den_row_vec.transpose() ;
    cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;

}
