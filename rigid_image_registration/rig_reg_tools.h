/*
Header file of rigid registration tools, rig_reg_tools.cpp

Author:
Nemanja Sakac, Student
Faculty of Technical Sciences, University of Novi Sad
Novi Sad, Serbia
Date: 2021.
*/

#pragma once

// ----------------------------------------------------------------------------
// LIBRARIES
// ----------------------------------------------------------------------------
#include <chrono>
using namespace std::chrono;

#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "shift.h"


// ----------------------------------------------------------------------------
// GLOBALS
// ----------------------------------------------------------------------------

// STRUCTURES
// ----------------------------------------------------------------------------

// Filter kernel parameters
typedef struct FiltParams
{
    int filt_width;
    double filt_std;
    double filt_thresh;
} FiltParams;

// Optimum parameters
typedef struct OptimumParams
{
    // Translation parameters along the axes
    double tx;
    double ty;
    // Sum of squared differences for the given optimum values
    double ssq;
    // Number of iterations to convergence
    int num_iter;
    // Optimization success
    bool is_success;
} OptimumParams;

// Additional optimization parameters
typedef struct AdditionalParams
{
    // Step to calculate the objective function for
    double step_dxy;
    // Number of iterations to convergence
    mutable int n_iter;
    // Maximum number of iterations to convergence
    int max_iter;

} AdditionalParams;

// Gradient values
typedef struct Gradient
{
    // Gradient along the x-axis
    double gx;
    // Gradient along the y-axis
    double gy;
    // Normalized gradient along the x-axis
    double norm_gx;
    // Normalized gradient along the y-axis
    double norm_gy;
} Gradient;

// SSQ point values - for the given optimum parameters and those around it
typedef struct SSQPoints
{
    double ssq;
    double ssq_x1;
    double ssq_x2;
    double ssq_y1;
    double ssq_y2;
} SSQPoints;

// Region of interest (ROI)
typedef struct ROI
{
    // Top left corner pixel coordinates
    int topY;
    int topX;
    // Bottom right corner pixel coordinates
    int bottomY;
    int bottomX;
} ROI;

// S current for conjugate-gradient descent
typedef struct SCur
{
    double s_cur_x;
    double s_cur_y;
} SCur;


// VARIABLES
// ----------------------------------------------------------------------------

// Default z-normalization parameters
extern const FiltParams DEFAULT_FILT;
// Default additional parameters
extern const AdditionalParams DEFAULT_ADD;
// Default gradient descent method
extern const std::string DEFAULT_GRAD_DESC_METHOD;

// ----------------------------------------------------------------------------
// FUNCTION PROTOTYPES
// ----------------------------------------------------------------------------

/*
Checks if images have been read successfully
*/
inline bool is_read_success(const cv::Mat& img)
{
    return img.empty() ? false : true;
}

/*
Checks if optimization of the cost function is successful. Takes a reference to 
the SSQPoints structure, which contains sums of squared differences of 
translation parameters at the "optimum point" and the points around it. Returns 
true if the optimum point is the local minimum.
*/
inline bool is_opt_success(const SSQPoints& ssq_points)
{
    return (ssq_points.ssq_x1 > ssq_points.ssq
        && ssq_points.ssq_x2 > ssq_points.ssq
        && ssq_points.ssq_y1 > ssq_points.ssq
        && ssq_points.ssq_y2 > ssq_points.ssq) ? true : false;
}

/*
cv::Mat z_norm(const cv::Mat &ref_img, const FiltParams& params) 

Normalizes (standardizes) local (regional) image values, given by "ref_image", to 
zero mean and unit variance. For significant amounts of noise, a threshold is 
defined for regulation. The function uses a square Gaussian filter for mean 
estimation, with filter parameteres given by "params".

The function returns a normalized image. "ref_img" is a reference to the input
image matrix. "params" is a structure of 3 parameters for defining the filter 
kernel. It contains 3 fields: 
    int filt_width - Width of the Gaussian filter kernel. Should be defined as odd. 
    Width defined as even will be made odd by adding 1 to the width value.
    double filt_std - Standard deviation of the LP Gaussian filter kernel.
    double filt_thresh - Minimum pixel value to be used if the normalized pixel value 
    is lower than this value.

Author:
Nemanja Sakac, Student
Faculty of Technical Sciences, University of Novi Sad
Novi Sad, Serbia
Date: 2021.
*/
cv::Mat z_norm(const cv::Mat& ref_img, const FiltParams& params = DEFAULT_FILT);

// ----------------------------------------------------------------------------

/*
Calculates the sum of squared differences between the reference image, img_ref, 
and the image for registration, img_reg. tx and ty are the translation 
parameters by which the image for registration is shifted along the x and y 
axis prior to calculating the sum of squared differences. The sum of squared 
differences is calculated for a predefined region of interest after translation.
*/
double get_ssqd(const cv::Mat& img_ref, const cv::Mat& img_reg,
    const double tx, const double ty, const ROI& roi);

/*
Calculates the sum of squared differences between the reference image, img_ref,
and the image for registration, img_reg. tx and ty are the translation
parameters by which the image for registration is shifted along the x and y
axis prior to calculating the sum of squared differences.
*/
double get_ssqd(const cv::Mat& img_ref, const cv::Mat& img_reg, const double tx, const double ty);

// ----------------------------------------------------------------------------
/*
Calculates sums of squared differences for optimum translation parameters, tx
and ty and parameters around it to check for convergence. Parameters around 
the translation parameters are calculated left, right, up and down, one step, 
dxy, away from the optimum translation parameters.
*/
SSQPoints get_ssq_points(const cv::Mat& img_ref, const cv::Mat& img_reg,
    double tx, double ty, const double dxy);

/*
Calculates sums of squared differences for optimum translation parameters, tx
and ty and parameters around it to check for convergence. Parameters around
the translation parameters are calculated left, right, up and down, one step,
dxy, away from the optimum translation parameters.Sums of squared differences
are calculated for a given region of interest in the image, roi.
*/
SSQPoints get_ssq_points(const cv::Mat& img_ref, const cv::Mat& img_reg,
    const double tx, const double ty, const double dxy, const ROI &roi);

// ----------------------------------------------------------------------------

/*
Returns gradient and normalized gradient values along the x axis and y axis. 
img_ref and img_reg are the reference image and image for registration, 
respectively. ssq_points is a structure with the sums of squared differences 
calculated at optimum translation parameter values and in the neighbourhood
of optimum parameters.
*/
Gradient get_gradient(const cv::Mat& img_ref, const cv::Mat& img_reg, 
    const SSQPoints& ssq_points);

// ----------------------------------------------------------------------------

/*
Optimizes the sum of squared differences between the reference image, img_ref, 
and the image for registration, img_reg. gradient stores the previously 
calculated gradient values and uses them to find new optimum translation 
parameters. roi contains the coordinates of the region of interest, n_iter 
contains the current iteration. method_name is an optional parameter used to 
select the method of gradient descent during the optimization process. Two 
methods can be used: 
"vanilla", ordinary gradient descent, the default method used if the 
method_name argument has not been passed to the function, and 
"polak-ribiere", based on the Polak-Ribiere nonlinear conjugate gradient descent 
method.
*/
void grad_desc(OptimumParams& opt_params, Gradient& gradient,
    const cv::Mat& img_ref, const cv::Mat& img_reg, const ROI &roi,
    const int n_iter, const std::string& method_name = DEFAULT_GRAD_DESC_METHOD);

// ----------------------------------------------------------------------------


/*
Finds the optimum translation parameter values for rigid image registration. 
The reference image and image for registration are read from the system based 
on the image names (with path) provided to the function parameters img_ref_name 
and img_reg_name, respectively. 

Optional parameters are method_name, filt_params and add_params. method_name, 
specifies the gradient descent method to be used for optimization. Available 
options are "vanilla" for ordinary ("vanilla") gradient descent and 
"polak-ribiere", based on the Polak-Ribiere nonlinear conjugate gradient descent 
method. filt_params are LP Gauss filter parameters for the z-normalization 
function (see z_norm for more details). add_params is a structure of "additional 
parameters" step_dxy, n_iter and max_iter. step_dxy is the step to take to  
calculate the sum of squared differences around the optimal values. n_iter 
is the current iteration. max_iter is the maximum number of iterations allowed 
before the optimization is considered a failure.

Author:
Nemanja Sakac, Student
Faculty of Technical Sciences, University of Novi Sad
Novi Sad, Serbia
Date: 2021.
*/
OptimumParams find_optimum(std::string& img_ref_name, std::string& img_reg_name, 
    const std::string& method_name = DEFAULT_GRAD_DESC_METHOD, const FiltParams& filt_params = DEFAULT_FILT, 
    const AdditionalParams& add_params = DEFAULT_ADD);
