/* 
KF mnobar@ethz.ch

*/
#include<iostream>
// these two files define the Kalman filter class
#include "myKalmanFilter.h"
#include "myKalmanFilter.cpp"

using namespace std;
using namespace Eigen;

int main() {
//    // discretization time step
//    double h=0.1;
    // measurement noise standard deviation
//    double noiseStd = 1;

    // define model: A,B,C matrices
    Matrix<double, 3, 3> A{{1, 0, 0},
                           {0, 1, 0},
                           {0, 0, 1}};
    Matrix<double, 3, 1> B{{0},
                           {1},
                           {0}};
    Matrix<double, 3, 3> C{{1, 0, 0},
                           {0, 1, 0},
                           {0, 0, 1}};
    // covariance matrix of the state estimation error P0- abbreviated as "state covariance matrix"
    Matrix<double, 3, 3> P0{{1, 0, 0},
                            {0, 4, 0},
                            {0, 0, 1}};

    // covariance matrix of the measurement noise
    Matrix<double, 3, 3> R{{4, 0, 0},
                           {0, 25, 0},
                           {0, 0, 4}};
    // covariance matrix of the state disturbance
    Matrix<double, 3, 3> Q{{1, 0, 0},
                           {0, 4, 0},
                           {0, 0, 1}};
    // guess of the initial state estimate
    Matrix<double, 3, 1> x0{{514},
                            {-270},
                            {101}};


// uncomment this part if you want to double check that the matrices are properly defined
    /*
       cout<<A<<endl<<endl;
       cout<<B<<endl<<endl;
       cout<<C<<endl<<endl;
       cout<<P0<<endl<<endl;
       cout<<Q<<endl<<endl;
       cout<<R<<endl<<endl;
       cout<<x0<<endl<<endl;
   */

    // in this column vector we store the output from the Python script that generates the noisy output
    MatrixXd x_measured;
    MatrixXd y_measured;
    MatrixXd z_measured;
    MatrixXd tVec_measured;
    // you can call the function openData() without creating the object of the class Kalman filter
    // since openData is a static function!
    x_measured = KalmanFilter::openData("/home/mahdi/Documents/kalman/myCode/myOutput_x_model_0.csv");
    y_measured = KalmanFilter::openData("/home/mahdi/Documents/kalman/myCode/myOutput_y_model_0.csv");
    z_measured = KalmanFilter::openData("/home/mahdi/Documents/kalman/myCode/myOutput_z_model_0.csv");
    tVec_measured = KalmanFilter::openData("/home/mahdi/Documents/kalman/myCode/tVec_measured_model_0.csv");
    //cout<<X_measured<<endl;
    int N = x_measured.rows();
    cout << N << endl; //for verification

    // max number of data samples
    // this number is used to initialize zero matrices that are used to store
    // and track the data such as estimates, covariance matrices, gain matrices, etc.
    // we want to make sure that this number is larger than the number of data samples
    // that are imported from "output.csv"
    unsigned int maxDataSamples = N + 1;

    // create the Kalman filter
    KalmanFilter KalmanFilterObject(A, B, C, Q, R, P0, x0, maxDataSamples);
    // these two scalars are used as inputs and outputs in the Kalman filter functions
    // input always stays zero since we do not have an input to our system
    // output is equal to the measured output and is adjusted
    MatrixXd u;
    u.resize(1, 1); //SISO system
//    outputY.resize(1, 1);//SISO system


    //cout<<N<<endl;
    int dt;
    // this is the MAIN KALMAN FILTER LOOP - predict and update (SISO system)
    for (int k_measured = 1; k_measured < N; k_measured++) {
        u(0, 0) = 0.0341;
//        outputY(0, 0) = x_measured(k_measured, 0);
        // cout<<output1<<endl;
        // predict the estimate
//        cout << tVec_measured(k_measured);
//        cout << tVec_measured(k_measured-1);

        dt = std::round(tVec_measured(k_measured)) - std::round(tVec_measured(k_measured - 1));
        KalmanFilterObject.B(1) = dt; //[ms]
        KalmanFilterObject.predictEstimate(u);
        KalmanFilterObject.B(1) = 1; //[ms]
        KalmanFilterObject.prediction_aheads(u, dt);
        // update the estimate
        KalmanFilterObject.updateEstimate(x_measured(k_measured), y_measured(k_measured), z_measured(k_measured));
    }

// save the data
// save data can be used to verify this implementation 
// the "myDriverCode.py" uses the a posteriori estimate in the file "estimatesAposteriori.csv"
// to compare Python and C++ implementations
    KalmanFilterObject.saveData("myCode/myEstimatesAposteriori_model_0.csv",
                                "myCode/myEstimatesAprioriFile_model_0.csv",
                                "myCode/myCovarianceAposterioriFile_model_0.csv",
                                "myCode/myCovarianceAprioriFile_model_0.csv",
                                "myCode/myGainMatricesFile_model_0.csv", "myCode/myErrorsFile_model_0.csv", "myCode/X_prediction_ahead_model_0.csv");


    return 0;
}