/*
Rigid image registration (translation) of X-ray lung images using vanilla
and nonlinear conjugate (Polak-Ribiere) gradient descent. The file contains the 
main function of the project presenting the functionality of the implemented 
solution.

Subject: Medical Image Processing
Author:
Nemanja Sakac, Student
Faculty of Technical Sciences, University of Novi Sad
Novi Sad, Serbia
Date: 2021.
*/


#include "rig_reg_tools.h"
#include "tests.h"


int main()
{
	// ---------------------------------------------------------------------------
	// BASIC FUNCTIONALITY
	// ---------------------------------------------------------------------------
	
	// Image names
	std::string img_ref_name = "F:/projects/rigid_image_registration_gradient_descent/c++/rigid_image_registration/img/REG_HE.png";
	std::string img_reg_name = "F:/projects/rigid_image_registration_gradient_descent/c++/rigid_image_registration/img/REG_LE_05.png";

	// Test reading, error handling, defining region of interest
	/*
	test_basic_func(img_ref_name, img_reg_name);
	*/

	// ---------------------------------------------------------------------------
	// IMAGE PROCESSING FUNCTIONS
	// ---------------------------------------------------------------------------

	// Use defined region of interest
	/*
	img_ref_name = "F:/projects/rigid_image_registration_gradient_descent/c++/rigid_image_registration/img/REG_HE.png";
	img_reg_name = "F:/projects/rigid_image_registration_gradient_descent/c++/rigid_image_registration/img/REG_LE.png";
	*/

	// Define image objects
	cv::Mat img_ref;
	cv::Mat img_reg;

	// Read images of interest
	img_ref = cv::imread(img_ref_name);
	img_reg = cv::imread(img_reg_name);

	// Convert pixel values to double
	cv::Mat float_img_ref;
	cv::Mat float_img_reg;
	img_ref.convertTo(float_img_ref, CV_64F);
	img_reg.convertTo(float_img_reg, CV_64F);

	// Z-NORMALIZATION
	// ---------------------------------------------------------------------------

	// Test z-normalization
	
	// test_znorm(float_img_ref);
	

	// INTERPOLATION
	// ---------------------------------------------------------------------------

	// Test interpolation
	/*
	test_interp(float_img_ref);
	*/

	// FINDING THE OPTIMUM VALUES
	// ---------------------------------------------------------------------------

	// Success
	// test_convergence_sgd(img_ref_name, img_reg_name);
	// Beware of exceptions for large translation values
	// test_convergence_pr(img_ref_name, img_reg_name);
	
	// Wait for key press to exit
	cv::waitKey();
	
}


