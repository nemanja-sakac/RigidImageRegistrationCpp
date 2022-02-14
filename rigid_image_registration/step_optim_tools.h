/*
Header file of step optimization tools

Author:
Nemanja Sakac, Student
Faculty of Technical Sciences, University of Novi Sad
Novi Sad, Serbia
Date: 2021.
*/


#pragma once


#include "rig_reg_tools.h"


/*
Updates the translation parameter
*/
inline double update_trans_param(double t, double k, double grad)
{
    return t + k * grad;
}

/*
Searches for the optimum step in the direction of the gradient (local optimum) 
using the golden section search algorithm. The implementation is slow since 
the optimum step is found indirectly based on the optimum translation parameters.

left_lim and right_lim define the left and right bound, respectively,  i.e. 
the interval of parameter values for which optimization is carried out. 
img_ref and img_reg are the reference and image for registration, respectively, 
needed to calculate the value of the objective function. tx and ty are current 
optimum translation parameters and grad_x and grad_y are current gradient 
values which determine the direction of the search.

Author: Nemanja Sakac, Student
Faculty of Technical Sciences, University of Novi Sad
Novi Sad, Serbia
Date: 2021.
*/
double golden_section(double left_lim, double right_lim, const cv::Mat& img_ref,
    const cv::Mat& img_reg, double tx, double ty, double grad_x, double grad_y);

/*
Searches for the optimum step in the direction of the gradient (local optimum)
using the golden section search algorithm. The implementation is slow since
the optimum step is found indirectly based on the optimum translation parameters.

left_lim and right_lim define the left and right bound, respectively,  i.e.
the interval of parameter values for which optimization is carried out.
img_ref and img_reg are the reference and image for registration, respectively,
needed to calculate the value of the objective function. tx and ty are current
optimum translation parameters. grad_x and grad_y are current gradient
values which determine the direction of the search. roi determines the region 
of interest in the image for which the objective function is calculated (so the 
border residue after interpolation doesn't affect the sum of squared 
differences).


Author: Nemanja Sakac, Student
Faculty of Technical Sciences, University of Novi Sad
Novi Sad, Serbia
Date: 2021.
*/
double golden_section(double left_lim, double right_lim, const cv::Mat& img_ref,
    const cv::Mat& img_reg, double tx, double ty, double grad_x, double grad_y, const ROI &roi);
