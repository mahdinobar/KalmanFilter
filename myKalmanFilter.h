// header file for the Kalman filter class
// Auhtor: Mahdi Nobar
#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include<string>
#include<Eigen/Dense>
using namespace Eigen;
using namespace std;
class KalmanFilter{

    public:
        // Default constructor - edit this later
        KalmanFilter();
        
        // Overloaded constructor
        // -  takes A,B,C, Q, R, P0, x0 matrices/vectors and assigns them to private variables
        //
        // -  the input argument "maxMeasurements" is used to initialize zero matrices 
        // "estimatesAposteriori","estimatesApriori", "covarianceAposteriori", "covarianceApriori"
        // "gainMatrices", and "errors" that are used to store appropriate quantities during the propagation 
        // of the Kalman filter equations
        // 
        // - the private variable "k" is set to zero. This variable is used to track the current iteration 
        // of the Kalman filter.
        KalmanFilter(MatrixXd A, MatrixXd B, MatrixXd C, 
                     MatrixXd Q, MatrixXd R, MatrixXd P0, 
                     MatrixXd x0, unsigned int maxSimulationSamples);
        
        // this member function updates the estimate on the basis of the measurement 
        // it computes the Kalman filter gain matrix
        // it computes the a posteriori estimate 
        // it computes the a posteriori covariance matrix
        void updateEstimate(double measurement_x, double measurement_y, double measurement_z);
        
        // this member function predicts the estimate on the basis of the external input
        // it computes the a priori estimate
        // it computes the a priori covariance matrix
        void predictEstimate(MatrixXd externalInput);
        
        // this member function is used to load the measurement data from the external CSV file 
        // the values are stored in the output matrix 
        // MatrixXd is an Eigen typdef for Matrix<double, Dynamic, Dynamic>
        static MatrixXd openData(string fileToOpen); // measurement data from csv column vector file (this is fed to the updateEstimate function)
        
        // this member function saves the stored date in the corresponding CSV files
        void saveData(string estimatesAposterioriFile, string estimatesAprioriFile, 
                      string covarianceAposterioriFile, string covarianceAprioriFile, 
                      string gainMatricesFile, string errorsFile) const;

    // MatrixXd is an Eigen typdef for Matrix<double, Dynamic, Dynamic> means matrix dimensions are not fixed and changing
    MatrixXd B;
private:

        // this variable is used to track the current time step k of the estimator 
        // after every measurement arrives, this variables is incremented for +1 
        unsigned int k;

        // m - input dimension, n- state dimension, r-output dimension 
        unsigned int m,n,r; 

        // MatrixXd is an Eigen typdef for Matrix<double, Dynamic, Dynamic> means matrix dimensions are not fixed and changing
	    MatrixXd A,C,Q,R,P0; // A,B,C,Q, R, and P0 matrices (Q and R are disturbance and measurement noise covariance matrices
	    MatrixXd x0;     // initial state
	    
        // this matrix is used to store the a posteriori estimates xk^{+} starting from the initial estimate 
        // note: the estimates are stored column wise in this matrix, starting from    
        // x0^{+}=x0 - where x0 is an initial guess of the estimate
        MatrixXd estimatesAposteriori; // [\hat{x}^{+}_{0:N}]
        
        // this matrix is used to store the a apriori estimates xk^{-} starting from x1^{-}
        // note: the estimates are stored column wise in this matrix, starting from x1^{-}   
        // That is, x0^{-} does not exist, that is, the matrix starts from x1^{-} 
        MatrixXd estimatesApriori; // [\hat{x}^{-}_{0:N}]
        
        // this matrix is used to store the a posteriori estimation error covariance matrices Pk^{+}
        // note: the matrix starts from P0^{+}=P0, where P0 is the initial guess of the covariance
        MatrixXd covarianceAposteriori; // [\hat{P}^{+}_{0:N}]
        
        
        // this matrix is used to store the a priori estimation error covariance matrices Pk^{-}
        // note: the matrix starts from P1^{-}, that is, P0^{-} does not exist
        MatrixXd covarianceApriori; // [P^{-}_{1:N}]
        
        // this matrix is used to store the gain matrices Kk
        MatrixXd gainMatrices; // [K_{1:N}]
         
        // this list is used to store prediction errors error_k=y_k-C*xk^{-}
        MatrixXd errors; // [e_{1:N}] , e_{k}=y_{k}-C_{k}*\hat{x}^{-}_{k}
   
};



#endif