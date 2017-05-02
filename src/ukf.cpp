#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  //P_ = MatrixXd(5, 5);
  P_ = MatrixXd::Identity(5,5);

  //Note - The process noise - std_a and std_yawdd are tunable params
  //Appropriate values need to be set inorder to achieve the desired RMSE

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.8;// 6.0;//4;// 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;// 0.2;// 0.5;// 30;

  //Meauremnt noise for lidar and radar - these are provided (ideally by manufacturer)
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;


  //set state dimension
  n_x_ = 5;
  //set augmented dimension
  n_aug_ = 7;
  // initialized predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  //define spreading parameter
  lambda_ = 3 - n_aug_;

  //set vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i<2 * n_aug_ + 1; i++) {  
	  //2n+1 weights
	  weights_(i) = 0.5 / (n_aug_ + lambda_);
  }

  previous_timestamp_ = 0;
  is_initialized_ = false;
  NIS_laser_ = 0.0;
  NIS_radar_ = 0.0;

  debug_flag_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
	if (!is_initialized_) {
		x_ = VectorXd(5);
		previous_timestamp_ = meas_package.timestamp_;

		if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			/**
			Convert radar from polar (pho, phi, pho_dot) to ctrv (px, py, v, si, si_dot) coordinates and initialize state.
			*/
			
			VectorXd z_meas = VectorXd(3);
			z_meas << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), meas_package.raw_measurements_(2);
			double pho = z_meas(0);
			double phi = z_meas(1);
			double px = pho * cos(phi);
			double py = pho * sin(phi);
			x_ << px, py, 0.0, 0.0, 0.0;
			is_initialized_ = true;
			return;
		}
		if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
			/**
			/**
			Convert lidar (px, py, vx, vy) to ctrv (px, py, v, si, si_dot) coordinates and initialize state.
			*/
			x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0.0, 0.0, 0.0;
			is_initialized_ = true;
			return;
		}
		// Laser / radar mode only...need re-initialization
		return;
	}

	//compute the time elapsed between the current and previous measurements
	double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
	previous_timestamp_ = meas_package.timestamp_;

	if(debug_flag_)
		cout << "x_ state is = " << std::endl << x_ << std::endl;

	//Get predicted sigma points
	Prediction(dt);
	if(debug_flag_)
		cout << "Xsig_pred_ = " << std::endl << Xsig_pred_ << std::endl;

	//Get the update state and covariance matrix
	Update(meas_package);
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
	//create augmented sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	
	//Generate the augmented sigma points
	AugmentedSigmaPoints(&Xsig_aug);
	if (debug_flag_)
		cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;
	
	//predict sigma points
	for (int i = 0; i< 2 * n_aug_ + 1; i++)
	{
		double minThresh = 0.001;
		double p_x = Xsig_aug(0, i);
		double p_y = Xsig_aug(1, i);
		double v = Xsig_aug(2, i);
		double yaw = Xsig_aug(3, i);
		double yawd = Xsig_aug(4, i);
		double nu_a = Xsig_aug(5, i);
		double nu_yawdd = Xsig_aug(6, i);

		//predicted state values
		double px_p, py_p;

		//avoid division by zero
		if (fabs(yawd) > minThresh) {
			px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
			py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
		}
		else {
			px_p = p_x + v*delta_t*cos(yaw);
			py_p = p_y + v*delta_t*sin(yaw);
		}

		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;

		//add noise
		px_p += 0.5*nu_a*delta_t*delta_t * cos(yaw);
		py_p += 0.5*nu_a*delta_t*delta_t * sin(yaw);
		v_p += nu_a*delta_t;
		yaw_p += 0.5*nu_yawdd*delta_t*delta_t;
		yawd_p += nu_yawdd*delta_t;

		//write predicted sigma point into right column
		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	}
}


/**
* Updates the state and the state covariance matrix using a laser/radar measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::Update(MeasurementPackage meas_package) {
	
	//set measurement dimension, laser can measure px and py, radar can measure pho, phi and theta
	int n_z;
	if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
		n_z = 3;
	else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
		n_z = 2;
	else
		return;

	//Get predicted mean and covariance
	//create vector for predicted state
	VectorXd x = VectorXd(n_x_);

	//create covariance matrix for prediction
	MatrixXd P = MatrixXd(n_x_, n_x_);

	PredictMeanAndCovariance(&x, &P);

	//create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
	VectorXd z_pred = VectorXd(n_z);
	MatrixXd S = MatrixXd(n_z, n_z);

	//create vector for incoming radar measurement
	VectorXd z = VectorXd(n_z);
	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);
	Tc.fill(0.0);
	if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
		PredictLidarMeasurement(&Zsig, &z_pred, &S);
		z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1);
		for (int i = 0; i < 2 * n_aug_ + 1; i++) { 
			VectorXd z_diff = Zsig.col(i) - z_pred;
			// state difference
			VectorXd x_diff = Xsig_pred_.col(i) - x;
			Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
		}

		//Kalman gain K;
		MatrixXd K = Tc * S.inverse();

		//residual
		VectorXd z_diff = z - z_pred;

		//update state mean and covariance matrix
		x_ = x + K * z_diff;
		P_ = P - K*S*K.transpose();

		//Compute NIS - Normalized innovation squared
		NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
	}
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		PredictRadarMeasurement(&Zsig, &z_pred, &S);
		z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), meas_package.raw_measurements_(2);
		for (int i = 0; i < 2 * n_aug_ + 1; i++) {
			VectorXd z_diff = Zsig.col(i) - z_pred;
			//angle normalization
			if (z_diff(1)> M_PI)
				z_diff(1) -= 2.*M_PI;
			if (z_diff(1)<-M_PI)
				z_diff(1) += 2.*M_PI;

			// state difference
			VectorXd x_diff = Xsig_pred_.col(i) - x;
			//angle normalization
			if (x_diff(3)> M_PI)
				x_diff(3) -= 2.*M_PI;
			if (x_diff(3)<-M_PI)
				x_diff(3) += 2.*M_PI;

			Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
		}
		//Kalman gain K;
		MatrixXd K = Tc * S.inverse();

		//residual
		VectorXd z_diff = z - z_pred;

		//angle normalization
		if (z_diff(1)> M_PI)
			z_diff(1) -= 2.*M_PI;
		if (z_diff(1)<-M_PI)
			z_diff(1) += 2.*M_PI;

		//update state mean and covariance matrix
		x_ = x + K * z_diff;
		P_ = P - K*S*K.transpose();
		//Compute NIS - Normalized innovation squared
		NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
	}

}



/**
* Computes the predicted mean and covariance using the predicted sigma points
* @param {VectorXd*} x_out
* @param {MatrixXd*} P_out
*/

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out) {

	//create vector for weights
	VectorXd weights = VectorXd(2 * n_aug_ + 1);

	//create vector for predicted state
	VectorXd x = VectorXd(n_x_);

	//create covariance matrix for prediction
	MatrixXd P = MatrixXd(n_x_, n_x_);

	//set weights
	double w0 = lambda_ / (lambda_ + n_aug_);
	double w1 = 1 / (2 * (lambda_ + n_aug_));
	//predict state mean
	x = w0*Xsig_pred_.col(0);
	for (int i = 1; i< 2 * n_aug_ + 1; i++)
	{
		x += w1*Xsig_pred_.col(i);
	}
	//predict state covariance matrix
	P = w0*(Xsig_pred_.col(0) - x)*(Xsig_pred_.col(0) - x).transpose();
	for (int i = 1; i< 2 * n_aug_ + 1; i++)
	{
		P += w1*(Xsig_pred_.col(i) - x)*(Xsig_pred_.col(i) - x).transpose();
	}

	//write result
	*x_out = x;
	*P_out = P;
}

/**
* Computes the predicted mean and covariance using the predicted sigma points in the lidar space
* @param {MatrixXd*} ZSig_out
* @param {VectorXd*} z_out
* @param {MatrixXd*} S_out
*/

void UKF::PredictLidarMeasurement(MatrixXd* Zsig_out, VectorXd* z_out, MatrixXd* S_out) {

	//set measurement dimension, laser can measure px, py
	int n_z = 2;

	//laser measurement noise standard deviation px in m
	double std_laspx = std_laspx_;

	//laser measurement noise standard deviation py in m
	double std_laspy = std_laspy_;

	//create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//transform sigma points into lidar measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		// measurement model
		Zsig(0, i) = p_x;                        
		Zsig(1, i) = p_y;                                 	
	}

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  
		VectorXd z_diff = Zsig.col(i) - z_pred;
		S += weights_(i) * z_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	MatrixXd R = MatrixXd(n_z, n_z);
	R << std_laspx*std_laspx, 0,
		0, std_laspy*std_laspy;
	S += R;

	//write result
	*Zsig_out = Zsig;
	*z_out = z_pred;
	*S_out = S;
}

/**
* Computes the predicted mean and covariance using the predicted sigma points in the radar space
* @param {MatrixXd*} ZSig_out
* @param {VectorXd*} z_out
* @param {MatrixXd*} S_out
*/

void UKF::PredictRadarMeasurement(MatrixXd* Zsig_out, VectorXd* z_out, MatrixXd* S_out) {

	//set measurement dimension, radar can measure r, phi, and r_dot
	int n_z = 3;

	//radar measurement noise standard deviation radius in m
	double std_radr = std_radr_;

	//radar measurement noise standard deviation angle in rad
	double std_radphi = std_radphi_;

	//radar measurement noise standard deviation radius change in m/s
	double std_radrd = std_radrd_;

	//create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; i++) { 
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);

		double v1 = cos(yaw)*v;
		double v2 = sin(yaw)*v;

		// measurement model
		Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                        //r
		Zsig(1, i) = atan2(p_y, p_x);                                 //phi
		Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
	}

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred += weights_(i) * Zsig.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) { 
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		S += weights_(i) * z_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	MatrixXd R = MatrixXd(n_z, n_z);
	R << std_radr*std_radr, 0, 0,
		0, std_radphi*std_radphi, 0,
		0, 0, std_radrd*std_radrd;
	S += R;

	//write result
	*Zsig_out = Zsig;
	*z_out = z_pred;
	*S_out = S;
}

/**
* Generates the augmented sigma points.
* @param {MatrixXd*} Xsig_out
*/
void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

	//create augmented mean vector
	VectorXd x_aug = VectorXd(7);

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(7, 7);

	//create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	//create augmented mean state
	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;

	//create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5, 5) = P_;

	//this is the augmented part
	P_aug(5, 5) = std_a_;// *std_a;
	P_aug(6, 6) = std_yawdd_;// *std_yawdd;

	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i< n_aug_; i++)
	{
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}
	
	//write result
	*Xsig_out = Xsig_aug;
}
