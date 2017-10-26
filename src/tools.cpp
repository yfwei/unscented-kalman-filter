#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  if (estimations.size() <= 0)
    return rmse;

  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size())
	return rmse;

  //accumulate squared residuals
  for(int i=0; i < estimations.size(); ++i){
	VectorXd e = estimations[i] - ground_truth[i];
	e = (e.array() * e.array());
	rmse += e;
  }

  //calculate the mean
  rmse /= estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}
