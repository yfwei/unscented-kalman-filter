#include "ukf.h"
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
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.8;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.4;

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

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_*std_radr_ , 0.0, 0.0,
       0.0, std_radphi_*std_radphi_, 0.0,
       0.0, 0.0, std_radrd_*std_radrd_;

  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;

  ///* State dimension
  n_x_ = 5;

  ///* Augmented state dimension
  n_aug_ = 7;

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  time_us_ = 0;

  is_initialized_ = false;

  // Create predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  weights_ = VectorXd::Zero(2 * n_aug_ + 1);
  weights_.setConstant(1.0 / (2 * (lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_ + n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0],
            meas_package.raw_measurements_[1],
            0.0, 0.0, 0.0;
    } else {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double px = rho * cos(phi);
      double py = rho * sin(phi);
      x_ << px, py, 0.0, 0.0, 0.0;
    }

    // Initialize the state covariance matrix
    P_ << 0.001, 0, 0, 0, 0,
          0, 0.001, 0, 0, 0,
          0, 0, 8, 0, 0,
          0, 0, 0, 0, 0,
          0, 0, 0, 0, 0;

    time_us_ = meas_package.timestamp_;

    is_initialized_ = true;
    return;
  }

  //compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  if (meas_package.sensor_type_== MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  } else {
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  MatrixXd Xsig_aug;
  GenerateSigmaPoints(&Xsig_aug);

  SigmaPointPrediction(Xsig_aug, delta_t);

  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  //set measurement dimension, LADAR can measure px and py directly
  int n_z = 2;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);

  //transform sigma points into measurement space
  for (int i = 0; i <= 2 * n_aug_; i++) {
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
  }

  //calculate mean predicted measurement
  for (int i = 0; i <= 2 * n_aug_; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z,n_z);

  //calculate measurement covariance matrix S
  for (int i = 0; i <= 2 * n_aug_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  S += R_laser_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  //calculate cross correlation matrix
  for (int i = 0; i <= 2 * n_aug_; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Normalized Innovation Squared (NIS)
  //std::cout << "LiDAR NIS=" << z_diff.transpose() * S.inverse() * z_diff << "\n";
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);

  //transform sigma points into measurement space
  for (int i = 0; i <= 2 * n_aug_; i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double vx = Xsig_pred_(2, i) * cos(Xsig_pred_(3, i));
    double vy = Xsig_pred_(2, i) * sin(Xsig_pred_(3, i));
    double rho = sqrt(px * px + py * py);
    double phi = atan2(py, px);
    double rho_dot = 0.0;

    if(fabs(rho) > 0.0001)
      rho_dot = (px * vx + py * vy) / rho;
    else
      std::cout << "Error - Division by Zero" << std::endl;

    Zsig(0, i) = rho;
    Zsig(1, i) = phi;
    Zsig(2, i) = rho_dot;
  }

  //calculate mean predicted measurement
  for (int i = 0; i <= 2 * n_aug_; i++) {
      z_pred += weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z,n_z);

  //calculate measurement covariance matrix S
  for (int i = 0; i <= 2 * n_aug_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    NormalizeAngle(z_diff(1));

    S += weights_(i) * z_diff * z_diff.transpose();
  }
  S += R_radar_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  //calculate cross correlation matrix
  for (int i = 0; i <= 2 * n_aug_; i++) {

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  //angle normalization
  NormalizeAngle(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Normalized Innovation Squared (NIS)
  //std::cout << "RADAR NIS=" << z_diff.transpose() * S.inverse() * z_diff << "\n";
}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {
  //create augmented mean vector
  VectorXd x_aug = VectorXd::Zero(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0.0;
  x_aug(6) = 0.0;

  //create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  //create augmented sigma points
  double scaling_factor = sqrt(lambda_ + n_aug_);

  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1) = x_aug + scaling_factor * A.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - scaling_factor * A.col(i);
  }

  *Xsig_out = Xsig_aug;

  return;
}

void UKF::SigmaPointPrediction(MatrixXd& Xsig_aug, double delta_t) {
  Xsig_pred_.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double velocity = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yaw_rate = Xsig_aug(4, i);
    double nu_acceleration = Xsig_aug(5, i);
    double nu_yaw_acceleration = Xsig_aug(6, i);

    Xsig_pred_(0, i) += Xsig_aug(0, i);
    Xsig_pred_(1, i) += Xsig_aug(1, i);
    Xsig_pred_(2, i) += Xsig_aug(2, i);
    Xsig_pred_(3, i) += Xsig_aug(3, i);
    Xsig_pred_(4, i) += Xsig_aug(4, i);

    if (fabs(yaw_rate) < 0.0001) {
     Xsig_pred_(0, i) += velocity * cos(yaw) * delta_t;
     Xsig_pred_(1, i) += velocity * sin(yaw) * delta_t;

    } else {
     Xsig_pred_(0, i) += (velocity / yaw_rate) * (sin(yaw + yaw_rate * delta_t) - sin(yaw));
     Xsig_pred_(1, i) += (velocity / yaw_rate) * (-cos(yaw + yaw_rate * delta_t) + cos(yaw));
    }

    Xsig_pred_(2, i) += 0.0;
    Xsig_pred_(3, i) += yaw_rate * delta_t;
    Xsig_pred_(4, i) += 0.0;

    VectorXd noise = VectorXd(5);
    noise(0) = 0.5 * delta_t * delta_t * cos(yaw) * nu_acceleration;
    noise(1) = 0.5 * delta_t * delta_t * sin(yaw) * nu_acceleration;
    noise(2) = delta_t * nu_acceleration;
    noise(3) = 0.5 * delta_t * delta_t * nu_yaw_acceleration;
    noise(4) = delta_t * nu_yaw_acceleration;

    Xsig_pred_.col(i) += noise;
  }
}

void UKF::PredictMeanAndCovariance() {
  //predict state mean
  x_.fill(0.0);
  for (int i = 0; i <= 2 * n_aug_; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  NormalizeAngle(x_(3));

  //predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i <= 2 * n_aug_; i++) {
    VectorXd X_diff = Xsig_pred_.col(i) - x_;

    NormalizeAngle(X_diff(3));

    P_ += weights_(i) * X_diff * X_diff.transpose();
  }
}

void UKF::NormalizeAngle(double& a) {
  while (a> M_PI) a-=2.*M_PI;
  while (a<-M_PI) a+=2.*M_PI;
}

