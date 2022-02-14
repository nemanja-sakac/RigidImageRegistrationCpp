/*
Rigid registration functions toolbox. 

Author:
Nemanja Sakac, Student
Faculty of Technical Sciences, University of Novi Sad
Novi Sad, Serbia
Date: 2021.
*/


// ----------------------------------------------------------------------------
// HEADERS
// ----------------------------------------------------------------------------
#include "rig_reg_tools.h"
#include "step_optim_tools.h"
#include "shift.h"


// ----------------------------------------------------------------------------
// GLOBALS
// ----------------------------------------------------------------------------

// Default kernel values
const int FILT_WIDTH = 20;
const double FILT_STD = 20;
const double FILT_THRESH = 5;

// Default additional initialization parameters
const double STEP_DXY = 1;
// Maximum number of iterations to convergence
const int MAX_ITER = 100;

// Z-normalization parameters
const FiltParams DEFAULT_FILT = { FILT_WIDTH, FILT_STD, FILT_THRESH };
// Default additional params
const AdditionalParams DEFAULT_ADD = { STEP_DXY, 1, MAX_ITER };
// Optimum parameters
OptimumParams INIT_OPT_PARAMS = { 0.0, 0.0, 0.0, 1, false };
// Default gradient descent method
const std::string DEFAULT_GRAD_DESC_METHOD = "vanilla";
// S current for conjugate-gradient descent
SCur s_cur = { 0.0, 0.0 };

// ----------------------------------------------------------------------------
// FUNCTION DEFINITIONS
// ----------------------------------------------------------------------------


OptimumParams find_optimum(std::string& img_ref_name, std::string& img_reg_name,
    const std::string &method_name, const FiltParams &filt_params, 
    const AdditionalParams &add_variables)
{
    // Read images of interest
    cv::Mat img_ref = cv::imread(img_ref_name);
    cv::Mat img_reg = cv::imread(img_reg_name);

    // Error handling if the images have not been read successfully
    if (!is_read_success(img_ref) || !is_read_success(img_reg))
    {
        if (!is_read_success(img_ref))
        {
            std::cout << "Reference image has not been read successfully" << '\n';
        }
        if (!is_read_success(img_reg))
        {
            std::cout << "Image for registration has not been read successfully" << '\n';
        }

        return INIT_OPT_PARAMS;
    }

    // Defining ROI
    ROI roi = { static_cast<int>(0.15 * img_ref.rows), 
        static_cast<int>(0.10 * img_ref.cols), 
        static_cast<int>(0.85 * img_ref.rows), 
        static_cast<int>(0.90 * img_ref.cols) };

    // Convert pixel values to double
    cv::Mat float_img_ref;
    cv::Mat float_img_reg;
    img_ref.convertTo(float_img_ref, CV_64F);
    img_reg.convertTo(float_img_reg, CV_64F);

    // Z-normalization
    float_img_ref = z_norm(float_img_ref, filt_params);
    float_img_reg = z_norm(float_img_reg, filt_params);

    // Initialize optimum parameters
    OptimumParams optimum_params = INIT_OPT_PARAMS;
    // Initialize additional parameters
    const AdditionalParams add_params = add_variables;
    
    while (!optimum_params.is_success 
        && (add_params.n_iter < add_params.max_iter))
    {

        double ssq = get_ssqd(float_img_ref, float_img_reg, optimum_params.tx, optimum_params.ty, roi);
        optimum_params.ssq = ssq;
        
        
        // Display the convergence rate
        // std::cout << std::setprecision(8) << optimum_params.tx << " " << optimum_params.ty << " " << optimum_params.ssq << '\n';
        
        
        // Get points around current optimum value
        SSQPoints ssq_points = get_ssq_points(float_img_ref, float_img_reg,
            optimum_params.tx, optimum_params.ty, add_params.step_dxy, roi);

        // Check if the optimization is successful
        // In case it is not successful (optimization diverges), it is going to return the last calculated parameters
        if (is_opt_success(ssq_points)
            || (optimum_params.tx > 100 || optimum_params.ty > 100)
            || (optimum_params.tx < -100 || optimum_params.ty < -100))
        {
            optimum_params.is_success = true;

            // Print optimum parameter values and the value of the sum of squared differences
            // std::cout << add_params.n_iter << std::endl;
            
            return optimum_params;
        }

        // Calculate gradient and normalized gradient
        Gradient gradient = get_gradient(float_img_ref, float_img_reg, ssq_points);
        
        // Perform gradient descent/search for optimum value
        grad_desc(optimum_params, gradient, float_img_ref, float_img_reg, roi, add_params.n_iter, method_name);

        // Update iteration
        ++add_params.n_iter;
        optimum_params.num_iter = add_params.n_iter;
    }

    // Did not converge
    return optimum_params;

}


SSQPoints get_ssq_points(const cv::Mat& img_ref, const cv::Mat& img_reg,
    const double tx, const double ty, const double dxy, const ROI& roi)
{
    // Calculate current ssqd
    double ssq = get_ssqd(img_ref, img_reg, tx, ty, roi);

    // Find 4 points in the area of optimum parameters to calculate the gradient
    // Combinations of optimum points are generated
    // Right
    double tx1 = tx + dxy;
    // Left
    double tx2 = tx - dxy;
    // Bottom
    double ty1 = ty + dxy;
    // Top
    double ty2 = ty - dxy;

    // Calculate ssqd for translation parameters defined by the new points
    // Right
    double ssq_x1 = get_ssqd(img_ref, img_reg, tx1, ty, roi);
    // Left
    double ssq_x2 = get_ssqd(img_ref, img_reg, tx2, ty, roi);
    // Bottom
    double ssq_y1 = get_ssqd(img_ref, img_reg, tx, ty1, roi);
    // Top
    double ssq_y2 = get_ssqd(img_ref, img_reg, tx, ty2, roi);

    SSQPoints ssq_points = SSQPoints{ ssq, ssq_x1, ssq_x2, ssq_y1, ssq_y2 };
    return ssq_points;
}


SSQPoints get_ssq_points(const cv::Mat& img_ref, const cv::Mat& img_reg,
    const double tx, const double ty, const double dxy)
{
    // Calculate current ssqd
    double ssq = get_ssqd(img_ref, img_reg, tx, ty);

    // Find 4 points in the area of optimum parameters to calculate the gradient
    // Combinations of optimum points are generated
    // Right
    double tx1 = tx + dxy;
    // Left
    double tx2 = tx - dxy;
    // Bottom
    double ty1 = ty + dxy;
    // Top
    double ty2 = ty - dxy;

    // Calculate ssqd for translation parameters defined by the new points
    // Right
    double ssq_x1 = get_ssqd(img_ref, img_reg, tx1, ty);
    // Left
    double ssq_x2 = get_ssqd(img_ref, img_reg, tx2, ty);
    // Bottom
    double ssq_y1 = get_ssqd(img_ref, img_reg, tx, ty1);
    // Top
    double ssq_y2 = get_ssqd(img_ref, img_reg, tx, ty2);

    SSQPoints ssq_points = SSQPoints{ ssq, ssq_x1, ssq_x2, ssq_y1, ssq_y2 };
    return ssq_points;
}


double get_ssqd(const cv::Mat& img_ref, const cv::Mat& img_reg, const double tx, const double ty)
{
    // Translate the image for registration
    // Define translation step
    cv::Point2f delta = cv::Point2f(tx, ty);
    // Shift (translate) image by defined step
    cv::Mat img_interp;
    shift(img_reg, img_interp, delta, cv::BORDER_CONSTANT);

    // Calculate objective function
    cv::Mat diff = img_ref - img_interp;
    cv::Mat diff_sq;
    cv::multiply(diff, diff, diff_sq);
    double ssqd = cv::sum(diff_sq)[0];
    return ssqd;
}


double get_ssqd(const cv::Mat& img_ref, const cv::Mat& img_reg, 
    const double tx, const double ty, const ROI& roi)
{
    // Translate the image for registration
    // Define translation step
    cv::Point2f delta = cv::Point2f(-tx, -ty);
    // Shift (translate) image by defined step
    cv::Mat img_interp;
    shift(img_reg, img_interp, delta, cv::BORDER_CONSTANT);

    // Define regions of interest
    cv::Mat img_ref_roi = img_ref(cv::Range(roi.topY, roi.bottomY),
        cv::Range(roi.topX, roi.bottomX));
    cv::Mat img_reg_roi = img_interp(cv::Range(roi.topY, roi.bottomY),
        cv::Range(roi.topX, roi.bottomX));

    // Calculate objective function
    cv::Mat diff = img_ref_roi - img_reg_roi;
    cv::Mat diff_sq;
    cv::multiply(diff, diff, diff_sq);
    double ssqd = cv::sum(diff_sq)[0];
    return ssqd;
}


Gradient get_gradient(const cv::Mat& img_ref, const cv::Mat& img_reg, 
    const SSQPoints& ssq_points)
{
    // Calculate the gradient in both the x and y direction
    double gx = ssq_points.ssq_x1 - ssq_points.ssq_x2;
    double gy = ssq_points.ssq_y1 - ssq_points.ssq_y2;

    // Normalize the gradient
    double denom = std::sqrt(std::pow(gx, 2) + std::pow(gy, 2));
    double norm_gx = gx / denom;
    double norm_gy = gy / denom;

    Gradient gradient = Gradient{ gx, gy, norm_gx, norm_gy };
    return gradient;
}


void grad_desc(OptimumParams& opt_params, Gradient& gradient,
    const cv::Mat& img_ref, const cv::Mat& img_reg, const ROI& roi,
    const int n_iter, const std::string& method_name)
{
    if (method_name == "vanilla")
    {
        // Find the optimum step in the gradient direction
        double opt_step = golden_section(-20, 20, img_ref, img_reg,
            opt_params.tx, opt_params.ty, gradient.norm_gx, gradient.norm_gy, roi);

        // Calculate new optimum parameters
        opt_params.tx += opt_step * gradient.norm_gx;
        opt_params.ty += opt_step * gradient.norm_gy;

    }
    else if (method_name == "polak-ribiere")
    {
        // Update current conjugate gradient direction
        if (n_iter == 1)
        {
            s_cur.s_cur_x = gradient.norm_gx;
            s_cur.s_cur_y = gradient.norm_gy;
        }
        else
        {
            // beta = (norm_g' * (norm_g - s_cur)) / (s_cur' * s_cur)
            double beta = (gradient.norm_gx * (gradient.norm_gx - s_cur.s_cur_x)
                + gradient.norm_gy * (gradient.norm_gy - s_cur.s_cur_y))
                / (std::pow(gradient.norm_gx, 2) + std::pow(gradient.norm_gy, 2));

            // Calculate new conjugate gradient direction
            s_cur.s_cur_x = gradient.norm_gx + beta * s_cur.s_cur_x;
            s_cur.s_cur_y = gradient.norm_gy + beta * s_cur.s_cur_y;

        }
        // Find the optimum step in the gradient direction
        double opt_step = golden_section(-20, 20, img_ref, img_reg,
            opt_params.tx, opt_params.ty, s_cur.s_cur_x, s_cur.s_cur_y, roi);

        // Calculate new optimum parameters
        opt_params.tx += opt_step * s_cur.s_cur_x;
        opt_params.ty += opt_step * s_cur.s_cur_y;
    }
    else
    {
        std::cout << "Gradient descent method " << "\"" << method_name << "\""
            << " does not exist." << std::endl;
    }
}


cv::Mat z_norm(const cv::Mat &ref_img, const FiltParams& params)
{
    // Kernel size container
    /*
    cv::Size kern_size = (params.filt_width % 2 != 0) ?
        cv::Size(params.filt_width, params.filt_width) : 
        cv::Size(params.filt_width + 1, params.filt_width + 1);
    */
    int kern_size = (params.filt_width % 2 != 0) ?
        params.filt_width : params.filt_width + 1;

    // Filter kernel standard deviation
    double filt_std = params.filt_std;
    // Filtering threshold
    double filt_thresh = params.filt_thresh;


    // Normalized image is the destination image
    cv::Mat loc_mean = cv::Mat(ref_img.rows, ref_img.cols, CV_64F);

    // Create Gaussian kernel
    cv::Mat gaussKernel = cv::getGaussianKernel(kern_size, filt_std, CV_64F);
    // Filter with Gaussian kernel
    cv::sepFilter2D(ref_img, loc_mean, CV_64F, gaussKernel, gaussKernel, cv::Point(-1, -1), 0, 0);
    
    /*
    // Estimate local mean by filtering
    cv::GaussianBlur(ref_img, loc_mean, kern_size, filt_std, filt_std, 0);
    */

    // Define local variance matrix
    cv::Mat loc_var = cv::Mat(ref_img.rows, ref_img.cols, CV_64F);

    // Calculate squared differences
    // Calculate differences
    loc_var = (ref_img - loc_mean);
    // Square differences
    loc_var = loc_var.mul(loc_var);

    // Estimate local variance
    /*
    cv::GaussianBlur(loc_var, loc_var, kern_size, filt_std, filt_std, 0);
    */
    cv::sepFilter2D(loc_var, loc_var, CV_64F, gaussKernel, gaussKernel, cv::Point(-1, -1), 0, 0);

    // Threshold image
    cv::max(loc_var, filt_thresh, loc_var);

    // Get square root of each pixel value
    cv::sqrt(loc_var, loc_var);

    // Normalize image
    loc_var = (ref_img - loc_mean) / loc_var;
    
	/*
    // Convert back to unsigned int
    cv::Mat norm_img;
    loc_var.convertTo(norm_img, CV_8UC1, 255, 0);
    */
	
    return loc_var;
}


