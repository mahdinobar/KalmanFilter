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
    bool model_0 = false;
    bool model_1 = true;
    // define model: A_KF,B_KF,C_KF matrices
    MatrixXd A_KF;
    MatrixXd B_KF;
    MatrixXd C_KF;
    // covariance matrix of the state estimation error P0_KF- abbreviated as "state covariance matrix"
    MatrixXd P0_KF;

    // covariance matrix of the measurement noise
    MatrixXd R_KF;
    // covariance matrix of the state disturbance
    MatrixXd Q_KF;
    // guess of the initial state estimate
    MatrixXd x0_KF;

    MatrixXd u_KF;

    // in this column vector we store the output from the Python script that generates the noisy output
    MatrixXd x_measured;
    MatrixXd y_measured;
    MatrixXd z_measured;
    MatrixXd tVec_measured;
    if (model_0 == true) {
        // define model: A_KF,B_KF,C_KF matrices
        Matrix<double, 3, 3> A_KF{{1, 0, 0},
                                  {0, 1, 0},
                                  {0, 0, 1}};
        Matrix<double, 3, 1> B_KF{{0},
                                  {1},
                                  {0}};
        Matrix<double, 3, 3> C_KF{{1, 0, 0},
                                  {0, 1, 0},
                                  {0, 0, 1}};
        // covariance matrix of the state estimation error P0_KF- abbreviated as "state covariance matrix"
        Matrix<double, 3, 3> P0_KF{{1, 0, 0},
                                   {0, 4, 0},
                                   {0, 0, 1}};

        // covariance matrix of the measurement noise
        Matrix<double, 3, 3> R_KF{{4, 0,  0},
                                  {0, 25, 0},
                                  {0, 0,  4}};
        // covariance matrix of the state disturbance
        Matrix<double, 3, 3> Q_KF{{1, 0, 0},
                                  {0, 4, 0},
                                  {0, 0, 1}};
        // guess of the initial state estimate
        Matrix<double, 3, 1> x0_KF{{514},
                                   {-270},
                                   {101}};

        x_measured = KalmanFilter::openData("/home/mahdi/Documents/kalman/myCode/myOutput_x_model_0.csv");
        y_measured = KalmanFilter::openData("/home/mahdi/Documents/kalman/myCode/myOutput_y_model_0.csv");
        z_measured = KalmanFilter::openData("/home/mahdi/Documents/kalman/myCode/myOutput_z_model_0.csv");
        tVec_measured = KalmanFilter::openData("/home/mahdi/Documents/kalman/myCode/tVec_measured_model_0.csv");

        u_KF.resize(1, 1); //SISO system
        u_KF(0, 0) = 0.0341;

    } else if (model_1 = true) {
        // define model: A_KF,B_KF,C_KF matrices
//        Matrix<double, 6, 6> A_KF{{1, 0, 0, 0, 0, 0},
//                               {0, 1, 0, 0, 0, 0},
//                               {0, 0, 1, 0, 0, 0},
//                               {0, 0, 0, 1, 0, 0},
//                               {0, 0, 0, 0, 1, 0},
//                               {0, 0, 0, 0, 0, 1}};
        A_KF.resize(6, 6);
        A_KF << 1, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0,
                0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 1;


//        Matrix<double, 6, 1> B_KF{{0},
//                               {0},
//                               {0},
//                               {0},
//                               {0},
//                               {0}};
        B_KF.resize(6, 1);
        B_KF << 0, 0, 0, 0, 0, 0;

//        Matrix<double, 3, 6> C_KF{{1, 0, 0, 0, 0, 0},
//                               {0, 1, 0, 0, 0, 0},
//                               {0, 0, 1, 0, 0, 0}};
        C_KF.resize(3, 6);
        C_KF << 1, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0;
        // covariance matrix of the state estimation error P0_KF- abbreviated as "state covariance matrix"
//        Matrix<double, 6, 6> P0_KF{{1, 0, 0, 0, 0, 0},
//                                {0, 4, 0, 0, 0, 0},
//                                {0, 0, 1, 0, 0, 0},
//                                {0, 0, 0, 1, 0, 0},
//                                {0, 0, 0, 0, 4, 0},
//                                {0, 0, 0, 0, 0, 1}};
        P0_KF.resize(6, 6);
        P0_KF << 1, 0, 0, 0, 0, 0,
                0, 4, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0,
                0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 4, 0,
                0, 0, 0, 0, 0, 1;

        // covariance matrix of the measurement noise
//        Matrix<double, 3, 3> R_KF{{1, 0, 0},
//                                  {0, 4, 0},
//                                  {0, 0, 1}};
        R_KF.resize(3, 3);
        R_KF << 1, 0, 0,
                0, 4, 0,
                0, 0, 1;
        // covariance matrix of the state disturbance
//        Matrix<double, 6, 6> Q_KF{{0, 0, 0, 0,     0,     0},
//                                  {0, 0, 0, 0,     0,     0},
//                                  {0, 0, 0, 0,     0,     0},
//                                  {0, 0, 0, 65e-3, 0,     0},
//                                  {0, 0, 0, 0,     13e-3, 0},
//                                  {0, 0, 0, 0,     0,     65e-3}};
        Q_KF.resize(6, 6);
        Q_KF<<0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 65e-3, 0, 0,
                0, 0, 0, 0, 13e-3, 0,
                0, 0, 0, 0, 0, 65e-3;
        // guess of the initial state estimate
//        Matrix<double, 6, 1> x0_KF{{514},
//                                   {-270},
//                                   {101},
//                                   {0},
//                                   {0.0341},
//                                   {0}};
        x0_KF.resize(6, 1);
        x0_KF<<514,-270,101,0,0.0341,0;

        x_measured = KalmanFilter::openData("/home/mahdi/Documents/kalman/myCode/myOutput_x_model_1.csv");
        y_measured = KalmanFilter::openData("/home/mahdi/Documents/kalman/myCode/myOutput_y_model_1.csv");
        z_measured = KalmanFilter::openData("/home/mahdi/Documents/kalman/myCode/myOutput_z_model_1.csv");
        tVec_measured = KalmanFilter::openData("/home/mahdi/Documents/kalman/myCode/tVec_measured_model_1.csv");

        u_KF.resize(1, 1); //SISO system
        u_KF(0, 0) = 0;

    }

    int N = x_measured.rows();

    unsigned int maxDataSamples = N + 1;

    // create the Kalman filter
    KalmanFilter KalmanFilterObject(A_KF, B_KF, C_KF, Q_KF, R_KF, P0_KF, x0_KF, maxDataSamples);

    int dt;
    // this is the MAIN KALMAN FILTER LOOP - predict and update (SISO system)
    for (int k_measured = 1; k_measured < N; k_measured++) {
        dt = std::round(tVec_measured(k_measured)) - std::round(tVec_measured(k_measured - 1));
        if (model_0 == true) {
            KalmanFilterObject.B(1) = dt; //[ms]
        } else if (model_1 == true) {
            KalmanFilterObject.A(0, 3) = dt; //[ms]
            KalmanFilterObject.A(1, 4) = dt; //[ms]
            KalmanFilterObject.A(2, 5) = dt; //[ms]
        }
        KalmanFilterObject.predictEstimate(u_KF);
        if (model_0 == true) {
            KalmanFilterObject.B(1) = 1; //[ms]
        } else if (model_1 == true) {
            KalmanFilterObject.A(0, 3) = 1; //[ms]
            KalmanFilterObject.A(1, 4) = 1; //[ms]
            KalmanFilterObject.A(2, 5) = 1; //[ms]
        }
        KalmanFilterObject.prediction_aheads(u_KF, dt);
        // update the estimate
        KalmanFilterObject.updateEstimate(x_measured(k_measured), y_measured(k_measured), z_measured(k_measured));
    }

// save the data
// save data can be used to verify this implementation 
// the "myDriverCode.py" uses the a posteriori estimate in the file "estimatesAposteriori.csv"
// to compare Python and C_KF++ implementations
    if (model_0 == true) {
        KalmanFilterObject.saveData("myCode/myEstimatesAposteriori_model_0.csv",
                                    "myCode/myEstimatesAprioriFile_model_0.csv",
                                    "myCode/myCovarianceAposterioriFile_model_0.csv",
                                    "myCode/myCovarianceAprioriFile_model_0.csv",
                                    "myCode/myGainMatricesFile_model_0.csv", "myCode/myErrorsFile_model_0.csv",
                                    "myCode/X_prediction_ahead_model_0.csv");
    } else if (model_1 == true) {
        KalmanFilterObject.saveData("myCode/myEstimatesAposteriori_model_1.csv",
                                    "myCode/myEstimatesAprioriFile_model_1.csv",
                                    "myCode/myCovarianceAposterioriFile_model_1.csv",
                                    "myCode/myCovarianceAprioriFile_model_1.csv",
                                    "myCode/myGainMatricesFile_model_1.csv", "myCode/myErrorsFile_model_1.csv",
                                    "myCode/X_prediction_ahead_model_1.csv");
        cout << "successfully ende!" << endl;
    }


    return 0;
}