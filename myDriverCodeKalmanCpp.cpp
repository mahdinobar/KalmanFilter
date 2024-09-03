/* 
KF mnobar@ethz.ch

*/
#include<iostream>
// these two files define the Kalman filter class
#include "myKalmanFilter.h"
#include "myKalmanFilter.cpp"

using namespace std;
using namespace Eigen;

int main()
{
    // discretization time step 
    double h=0.1;
    // measurement noise standard deviation
    double noiseStd=1;

    // define model: A,B,C matrices
    Matrix <double,3,3> A {{1,h,0.5*(h*h)},{0, 1, h},{0,0,1}};
    Matrix <double,3,1> B {{0},{0},{0}}; 
    Matrix <double,1,3> C {{1,0,0}};
    // covariance matrix of the state estimation error P0- abbreviated as "state covariance matrix"
    MatrixXd P0; P0.resize(3,3); P0= MatrixXd::Identity(3,3);
    // covariance matrix of the measurement noise
    Matrix <double,1,1> R {{noiseStd*noiseStd}};
    // covariance matrix of the state disturbance
    MatrixXd Q; Q.resize(3,3); Q.setZero(3,3);
    // guess of the initial state estimate
    Matrix <double,3,1> x0 {{0},{0},{0}};


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
    MatrixXd outputNoisy;
    // you can call the function openData() without creating the object of the class Kalman filter
    // since openData is a static function!
    outputNoisy=KalmanFilter::openData("myCode/myOutput_x.csv");
    //cout<<outputNoisy<<endl;
    int sampleNumber=outputNoisy.rows();
    cout<<sampleNumber<<endl; //for verification
    
    // max number of data samples 
    // this number is used to initialize zero matrices that are used to store 
    // and track the data such as estimates, covariance matrices, gain matrices, etc.
    // we want to make sure that this number is larger than the number of data samples 
    // that are imported from "output.csv"
    unsigned int maxDataSamples=sampleNumber+10;

    // create the Kalman filter
    KalmanFilter KalmanFilterObject(A, B, C, Q, R, P0, x0, maxDataSamples);
    // these two scalars are used as inputs and outputs in the Kalman filter functions
    // input always stays zero since we do not have an input to our system
    // output is equal to the measured output and is adjusted
    MatrixXd inputU, outputY;
    inputU.resize(1,1); //SISO system
    outputY.resize(1,1);//SISO system
 

    //cout<<sampleNumber<<endl;

    // this is the MAIN KALMAN FILTER LOOP - predict and update (SISO system)
    for (int index1=0; index1<sampleNumber; index1++)
    {
        inputU(0,0)=0;
        outputY(0,0)=outputNoisy(index1,0);
        // cout<<output1<<endl;
        // predict the estimate
        KalmanFilterObject.predictEstimate(inputU);
        // update the estimate
        KalmanFilterObject.updateEstimate(outputY);
    }

// save the data
// save data can be used to verify this implementation 
// the "myDriverCode.py" uses the a posteriori estimate in the file "estimatesAposteriori.csv"
// to compare Python and C++ implementations
KalmanFilterObject.saveData("myCode/myEstimatesAposteriori.csv", "myCode/myEstimatesAprioriFile.csv",
                            "myCode/myCovarianceAposterioriFile.csv", "myCode/myCovarianceAprioriFile.csv",
                            "myCode/myGainMatricesFile.csv", "myCode/myErrorsFile.csv");


return 0;
}