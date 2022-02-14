/*
Step optimization tools, i.e. line search algorithm function implementations.
Currently contains only the golden section search algorithm.

Author:
Nemanja Sakac, Student
Faculty of Technical Sciences, University of Novi Sad
Novi Sad, Serbia
Date: 2021.
*/


#include "rig_reg_tools.h"
#include "step_optim_tools.h"


double golden_section(double left_lim, double right_lim, const cv::Mat& img_ref, 
    const cv::Mat& img_reg, double tx, double ty, double grad_x, double grad_y)
{

    // Maximum number of iterations
    int n_max = 50;
    // Tolerance for stopping the algorithm
    double epsilon = 1.0e-6;
    // Golden section coefficient
    double const tau = (3 - std::sqrt(5)) / 2;

    // Left and right bound of the interval 
    double a = left_lim;
    double b = right_lim;

    // Initialize number of iterations
    int n_iter = 0;

    // Calculate first interval
    double y = a + tau * (b - a);
    double z = a + (1 - tau) * (b - a);

    // Translation parameters
    double tx1 = update_trans_param(tx, y, grad_x);
    double ty1 = update_trans_param(ty, y, grad_y);
    double tx2 = update_trans_param(tx, z, grad_x);
    double ty2 = update_trans_param(ty, z, grad_y);

    double f_y = get_ssqd(img_ref, img_reg, tx1, ty1);
    double f_z = get_ssqd(img_ref, img_reg, tx2, ty2);

    while ((abs(b - a) >= epsilon) && (n_iter < n_max))
    {
        // Update iteration
        ++n_iter;
        if (f_y < f_z)
        {
            // New b is z
            // New a is still previous a
            b = z;
            z = y;

            tx2 = update_trans_param(tx, z, grad_x);
            ty2 = update_trans_param(ty, z, grad_y);
            f_z = get_ssqd(img_ref, img_reg, tx2, ty2);

            // New y
            y = a + tau * (b - a);
            tx1 = update_trans_param(tx, y, grad_x);
            ty1 = update_trans_param(ty, y, grad_y);
            f_y = get_ssqd(img_ref, img_reg, tx1, ty1);
        }
        else
        {
            a = y;
            y = z;
            tx1 = update_trans_param(tx, y, grad_x);
            ty1 = update_trans_param(ty, y, grad_y);
            f_y = get_ssqd(img_ref, img_reg, tx1, ty1);

            z = a + (1 - tau) * (b - a);
            tx2 = update_trans_param(tx, z, grad_x);
            ty2 = update_trans_param(ty, z, grad_y);
            f_z = get_ssqd(img_ref, img_reg, tx2, ty2);
        }        
    }

    // Value at minimum
    /*
    if (f_y < f_z)
    {
        f_x = f_y;
    }
    else
    {
        f_x = f_z;
    }
    */

    // Minimum found
    return y;

}

double golden_section(double left_lim, double right_lim, const cv::Mat& img_ref,
    const cv::Mat& img_reg, double tx, double ty, double grad_x, double grad_y, const ROI& roi)
{

    // Maximum number of iterations
    int n_max = 50;
    // Tolerance for stopping the algorithm
    double epsilon = 0.000001;
    // Golden section coefficient
    double const tau = (3 - std::sqrt(5)) / 2;

    // Left and right bound of the interval 
    double a = left_lim;
    double b = right_lim;

    // Initialize number of iterations
    int n_iter = 0;

    // Calculate first interval
    double y = a + tau * (b - a);
    double z = a + (1 - tau) * (b - a);

    // Translation parameters
    double tx1 = update_trans_param(tx, y, grad_x);
    double ty1 = update_trans_param(ty, y, grad_y);
    double tx2 = update_trans_param(tx, z, grad_x);
    double ty2 = update_trans_param(ty, z, grad_y);

    double f_y = get_ssqd(img_ref, img_reg, tx1, ty1, roi);
    double f_z = get_ssqd(img_ref, img_reg, tx2, ty2, roi);

    while ((abs(b - a) >= epsilon) && (n_iter < n_max))
    {
        // Update iteration
        ++n_iter;
        if (f_y < f_z)
        {
            // New b is z
            // New a is still previous a
            b = z;
            z = y;

            tx2 = update_trans_param(tx, z, grad_x);
            ty2 = update_trans_param(ty, z, grad_y);
            f_z = get_ssqd(img_ref, img_reg, tx2, ty2, roi);

            // New y
            y = a + tau * (b - a);
            tx1 = update_trans_param(tx, y, grad_x);
            ty1 = update_trans_param(ty, y, grad_y);
            f_y = get_ssqd(img_ref, img_reg, tx1, ty1, roi);
        }
        else
        {
            a = y;
            y = z;
            tx1 = update_trans_param(tx, y, grad_x);
            ty1 = update_trans_param(ty, y, grad_y);
            f_y = get_ssqd(img_ref, img_reg, tx1, ty1, roi);

            z = a + (1 - tau) * (b - a);
            tx2 = update_trans_param(tx, z, grad_x);
            ty2 = update_trans_param(ty, z, grad_y);
            f_z = get_ssqd(img_ref, img_reg, tx2, ty2, roi);
        }
    }

    // Value at minimum
    /*
    if (f_y < f_z)
    {
        f_x = f_y;
    }
    else
    {
        f_x = f_z;
    }
    */

    // Minimum found
    return y;

}


