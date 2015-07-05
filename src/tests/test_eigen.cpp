#include "linear.hpp"
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include "Eigen/Dense"
using namespace oplin;
using namespace std;
using namespace Eigen;
float calculate(const float coefficient,const float basePrice,
                const float eSuccRate,const float pt_real, const float et)
{
    unsigned int m_segment=5;
    unsigned int m_kxWindow=5;
    unsigned int m_avgEtWindow=50;
    VectorXf m_K = Eigen::VectorXf::Zero(4);
    m_K(0)=1.0;
    m_K(1)=0.8;
    m_K(2)=0.;
    m_K(3)=0.;
    MatrixXf m_X = Eigen::MatrixXf::Zero(m_kxWindow,4);// samples X
    VectorXf m_Y = Eigen::VectorXf::Zero(m_kxWindow);  // ÁÐÏòÁ¿Y
    float m_sumEt=0.;
    unsigned long m_count=0;

    if(pt_real <= 0.)
    {
        return 0.;
    }
    unsigned int index =  m_count % m_kxWindow;

    Eigen::VectorXf CurV = Eigen::VectorXf::Zero(4);
    if(m_count< 4)
    {
        m_X(index,0) = basePrice;
        m_X(index,1) = et;
        m_sumEt+=et;
        m_X(index,2)=m_sumEt/(index+1);
        if(index > 0)
        {
            m_X(index,3)=et-m_X(index-1,1);
        }
        else
        {
            m_X(index,3)=et;
        }
    }
    else
    {
        m_X(index,0) = basePrice;
        m_sumEt -= m_X(index,1);
        m_X(index,1) = et;
        m_sumEt += m_X(index,1);

        if(m_count < m_kxWindow)
        {
            m_X(index,2) = m_sumEt/(index+1);
        }
        else
        {
            m_X(index,2) = m_sumEt/m_kxWindow;
        }

        //det
        unsigned int lastIndex = index==0?m_kxWindow-1:index-1;
        m_X(index,3)=m_X(index,1) - m_X(lastIndex,1);
    }
    m_Y(index)=pt_real + basePrice*eSuccRate*0.15;

    CurV(0)=m_X(index,0);
    CurV(1)=m_X(index,1);
    CurV(2)=m_X(index,2);
    CurV(3)=m_X(index,3);
    m_count++;
    if((m_count >= 4))
    {
            std::cout<<"X:"<<m_X<<std::endl;
            std::cout<<"Y:"<<m_Y<<std::endl;
            Eigen::MatrixXf XT  = m_X.transpose();
            std::cout<<"XT:"<<XT<<std::endl;
            Eigen::VectorXf XTY = XT*m_Y;
            std::cout<<"XTY:"<<XTY<<std::endl;
            Eigen::MatrixXf XTX = Eigen::MatrixXf::Zero(4,4);
            Eigen::MatrixXf I   = Eigen::MatrixXf::Identity(4,4);
            XTX=XT*m_X;
            XTX=XTX+coefficient*I;
            XTX=XTX.inverse();
            std::cout<<"XTY.I:"<<XTX<<std::endl;
            m_K = XTX*XTY;
            std::cout<<"K:"<<m_K<<std::endl;
            //unsigned long max
            if(m_count >=4294967295)
                m_count = m_kxWindow;
    }
    float ptNext = CurV.transpose()*m_K;
    //ptNext = ptNext > basePrice?basePrice:ptNext;
    if (ptNext<=0.0)
    {
        if(m_Y(index) <= 0.0)
        {
            ptNext = basePrice;
            m_Y(index) = basePrice;
        }
        else
        {
            ptNext = m_Y(index);
        }
    }
    if (ptNext>150.0)
    {
        ptNext = 150.0;
    }
    return ptNext;
    /*
    m_matrixSumEt[segment].push_back(et);
    std::vector<float> tmp;

    sumet=calculateSumEt(segment);
    tmp.push_back(basePrice);
    tmp.push_back(eSuccRate);
    tmp.push_back(realSuccRate);
    tmp.push_back(pt);
    tmp.push_back(pt_real);
    m_matrixKx[segment].push_back(tmp);

    if((m_isFirst && m_matrixKx[segment].size() > 3) || m_matrixKx[segment].size() >= m_kxWindow)
    {
            Eigen::MatrixXf X = Eigen::MatrixXf::Zero(m_matrixKx[segment].size() - 1,3);// samples X
            Eigen::VectorXf Y = Eigen::VectorXf::Zero(m_matrixKx[segment].size() - 1);  // ÁÐÏòÁ¿Y

            std::vector<float> last;

            std::list<std::vector<float> >::iterator it = m_matrixKx[segment].begin();
            it++;
            int i = 1;
            for(;it != m_matrixKx[segment].end();it++)
            {
                last     = *it;
                std::vector<float> tmpBase = *(it--);
                it++;
                Y(i-1)   = last[4] - tmpBase[3];
                for(int j=0;j<3;j++)
                {
                    X(i-1,j) = last[j];
                }
                i++;
            }
            // bridge
            Eigen::MatrixXf XT  = X.transpose();
            Eigen::VectorXf XTY = XT*Y;
            Eigen::MatrixXf XTX = Eigen::MatrixXf::Zero(3,3);
            Eigen::MatrixXf I   = Eigen::MatrixXf::Identity(3,3);
            XTX=XT*X;
            XTX=XTX+coefficient*I;
            XTX=XTX.reverse();
            K = XTX*XTY;

            m_k1 = K(0);
            m_k2 = K(1);
            m_k3 = K(2);
            LOG(INFO,"%f %f %f\n",m_k1,m_k2,m_k3);
            m_matrixKx[segment].clear();
            m_matrixKx[segment].push_back(last);
    }
    return 0;*/
}
int main()
{
    MatrixXf m(3,3);
    m << 1,2,3,4,5,6,7,8,9;
    m /= 1.5;
    cout << m << endl;
    SpColVector spv(1000);
    SpRowMatrix spm(1000,1000);
    SpColMatrix spm_col(1000,1000);
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

    SpColVector copy_spv =
    spv.coeffRef(10) = 1.5;
    spv.coeffRef(999) = 1.5;
    spv.coeffRef(100) = 2.5;
    cout << copy_spv << endl;
    cout << spv << endl;
    cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    cout << spv.squaredNorm() / 2.0 << endl;
    for(size_t i = 0; i < 15 ;++i)
    {
        spm.insert(i,i*i) = 1.5;
        spm.insert(i+5,i*i) = 1.5;
        spm_col.insert(i,i*i) = 1.5;
        spm_col.insert(i+5,i*i) = 1.5;
    }
    SpColMatrix spm1(10,10);
    spm1.insert(9,9) = 101.5;
    spm1.insert(6,9) = 105.5;
    spm1.insert(8,8) = 1000.5;
    spm1.makeCompressed();
    cout << spm1 << endl;
    int a[] = {3,2,1,5,6,9,8,7,4,0};
    PermutationMatrix<Dynamic,Dynamic> perm_matrix(10);
    start = clock();
    for(int i =0 ;i < 10 ;i++)
    {
        perm_matrix.indices()[i] = a[i];
    }
    // perm_matrix.setIdentity();
    // perm_matrix.indices()[8] = 9;
    // perm_matrix.indices()[9] = 8;
    cout << perm_matrix.indices() << endl;

    spm1=  (spm1* perm_matrix).eval();
    cout << spm1 << endl;
    cout << "time permutation:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    cout << "swap test : " << spm1.coeffRef(9,8) << "," <<
        spm1.coeffRef(8,9) << endl;
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
    SpRowVector res = spv.transpose() * spm;
    cout << res.cols() << "," << res.rows()  << endl;
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
    cout << res1 << endl;
    cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    start = clock();
    RowVector res2 =  spm * den_row_vec.transpose() ;
    cout << "time test:" << float(clock() -start)/CLOCKS_PER_SEC << endl;
    cout << res2  <<endl;
    cout << res2 * 0.0001 <<endl;

    // Sparse by Dense Test
    cout << ">>>>>>>>>>Sparse Matrix By Dense Vector<<<<<<<<<<"<<endl;
    SpColMatrix l_m(10,10);
    for(size_t i = 0;i < 10;++i)
        for(size_t j=0;j < 10; ++j)
            l_m.coeffRef(i,j) = i*j;
    ColVector r_v = ColVector::Ones(10,1);
    cout << "l_m" << endl;
    cout << l_m << endl;
    cout << "r_v" << endl;
    cout << r_v << endl;
    ColVector result_mv = l_m * r_v;
    cout << "result" << endl;
    cout << result_mv << endl;

    // // Random Test
    // cout << ">>>>>>>>>>Random Test<<<<<<<<<<"<<endl;

    // for(int i = 0; i < 5; i++) {
    //     srand((unsigned int) time(0));
    //     std::this_thread::sleep_for (std::chrono::seconds(1));
    //     ColMatrix A = ColMatrix::Random(3, 3) / 2;
    //     cout << A <<endl;
    // }
    cout << "reached" << endl;
    // out of bound test
    MatrixXf oob_x = MatrixXf::Zero(4,4);
    oob_x(1,1) = 10000000000007.5;
    oob_x = oob_x + MatrixXf::Identity(4,4) * 0.01;
    cout << oob_x.inverse() << endl;

    // cout << oob_x(4,1);
    float result = calculate(0.01,100,0.7,10, 0.1);
    cout << "result: " << result <<endl;

}
