/*
Contains test functions for the rigid registration project.

Author:
Nemanja Sakac, Student
Faculty of Technical Sciences, University of Novi Sad
Novi Sad, Serbia
Date: 2021.
*/



#include "rig_reg_tools.h"



void test_znorm(const cv::Mat& img)
{
	/*
	* TEST 1 - Default paramater values
	*/
	// Default parameter values
	cv::Mat norm_image1 = z_norm(img);
	cv::namedWindow("Z-normalized Image 1");
	cv::imshow("Z-normalized Image 1", norm_image1);

	// Converting the image to UINT8
	cv::Mat norm_img1_UINT8;
	norm_image1.convertTo(norm_img1_UINT8, CV_8U, 255, 0);

	// Save resulting image
	cv::imwrite("znorm_img1.png", norm_img1_UINT8);


	/*
	* TEST 2 - New parameter values 
	*/
	FiltParams new_params = { 30, 20, 10 };
	cv::Mat norm_image2 = z_norm(img, new_params);
	cv::namedWindow("Z-normalized Image 2");
	cv::imshow("Z-normalized Image 2", norm_image2);

	// Converting the image to UINT8
	cv::Mat norm_img2_UINT8;
	norm_image2.convertTo(norm_img2_UINT8, CV_8U, 255, 0);

	// Save resulting image
	cv::imwrite("znorm_img2.png", norm_img2_UINT8);

}

void test_interp(cv::Mat &img)
{
	/*
	* TEST 1 - Testing with double values
	*/

	// Define step size
	double tx = 10;
	double ty = -45.3;

	// Z-normalize image
	cv::Mat norm_image1 = z_norm(img);

	// Interpolate image
	// Define translation step
	cv::Point2f delta = cv::Point2f(tx, ty);
	// Shift (translate) image by defined step
	cv::Mat interp_img1;
	shift(norm_image1, interp_img1, delta, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));

	cv::namedWindow("Interpolated Image 1");

	// Show resulting image
	cv::imshow("Interpolated Image 1", interp_img1);
}

/* 
* Tests basic functionality such as reading the image, error handling and 
* defining the region of interest (ROI)
*/
void test_basic_func(std::string& img_ref_name, std::string& img_reg_name)
{

	// Define an image object
	cv::Mat img_ref;
	cv::Mat img_reg;

	// Read images of interest
	img_ref = cv::imread(img_ref_name);
	img_reg = cv::imread(img_reg_name);

	// Error handling if the images have not been read successfully
	if (!is_read_success(img_ref) || !is_read_success(img_reg))
	{
		return;
	}

	// Show image sizes
	std::cout << "Reference image dimensions: " << img_ref.rows << " x " << img_ref.cols
		<< std::endl;
	std::cout << "Registration image dimensions: " << img_reg.rows << " x " << img_reg.cols
		<< std::endl;

	// Define the windows
	cv::namedWindow("Reference Image");
	cv::namedWindow("Registration Image");
	// Show the images
	cv::imshow("Reference Image", img_ref);
	cv::imshow("Registration Image", img_reg);

	// Region of Interest
	// Top left corner coordinates
	int topY = static_cast<int>(0.15 * img_ref.rows);
	int topX = static_cast<int>(0.10 * img_ref.cols);
	// Image height and width
	int bottomY = static_cast<int>(0.85 * img_ref.rows);
	int bottomX = static_cast<int>(0.90 * img_ref.cols);
	// Define regions of interest
	cv::Mat img_ref_roi = img_ref(cv::Range(topY, bottomY),
		cv::Range(topX, bottomX));
	cv::Mat img_reg_roi = img_reg(cv::Range(topY, bottomY),
		cv::Range(topX, bottomX));

	// Show defined regions of interest
	cv::namedWindow("ROI of Reference Image");
	cv::namedWindow("ROI of Registration Image");
	cv::imshow("ROI of Reference Image", img_ref_roi);
	cv::imshow("ROI of Registration Image", img_reg_roi);

	cv::imwrite("img_ref.png", img_ref_roi);
	cv::imwrite("img_reg.png", img_reg_roi);

}


/*
Steepest Gradient Descent test
*/
void test_convergence_sgd(std::string& img_ref_name, std::string& img_reg_name)
{

	// Initialize optimum parameter structure
	OptimumParams optimum_params;
	// Delimiter for displaying the results
	std::string result_delimiter = ", ";
	// Declare filter parameter structure
	FiltParams filt_params;


	// Create parameter combinations for testing
	std::vector<int> gauss_width = { 5, 10, 20, 30 };
	std::vector<double> gauss_sd = { 5, 10, 20 };
	std::vector<double> gauss_thresh = { 1, 5, 10 };


	std::cout << "Steepest Gradient Descent\n";
	for (int width : gauss_width)
	{
		for (double sd : gauss_sd)
		{
			for (double thresh : gauss_thresh)
			{
				// Create FiltParam structure
				filt_params.filt_width = width;
				filt_params.filt_std = sd;
				filt_params.filt_thresh = thresh;

				// Get starting timepoint
				auto start_sgd = high_resolution_clock::now();
				optimum_params = find_optimum(img_ref_name, img_reg_name, "vanilla", filt_params);
				auto stop_sgd = high_resolution_clock::now();

				// Get duration. Substart timepoints to  get duraTion. To cast it to proper unit
				// use duration cast method.
				auto duration_sgd = duration_cast<milliseconds>(stop_sgd - start_sgd);

				// Display results
				std::cout << width << result_delimiter
					<< sd << result_delimiter
					<< thresh << result_delimiter
					<< optimum_params.tx << result_delimiter
					<< optimum_params.ty << result_delimiter
					<< optimum_params.ssq << result_delimiter
					<< optimum_params.num_iter << result_delimiter
					<< duration_sgd.count() << "\n";

			}
		}
	}
}


/*
Polak-Ribiere test
*/
void test_convergence_pr(std::string & img_ref_name, std::string & img_reg_name)
{

	// Initialize optimum parameter structure
	OptimumParams optimum_params;
	// Delimiter for displaying the results
	std::string result_delimiter = ", ";
	// Declare filter parameter structure
	FiltParams filt_params;


	// Create parameter combinations for testing
	std::vector<int> gauss_width = { 5, 10, 20, 30 };
	std::vector<double> gauss_sd = { 5, 10, 20 };
	std::vector<double> gauss_thresh = { 1, 5, 10 };


	std::cout << "Polak-Ribiere\n";
	for (int width : gauss_width)
	{
		for (double sd : gauss_sd)
		{
			for (double thresh : gauss_thresh)
			{
				// Create FiltParam structure
				filt_params.filt_width = width;
				filt_params.filt_std = sd;
				filt_params.filt_thresh = thresh;

	
				// Get starting timepoint
				auto start_pr = high_resolution_clock::now();
				optimum_params = find_optimum(img_ref_name, img_reg_name, "polak-ribiere", filt_params);
				auto stop_pr = high_resolution_clock::now();

				// Get duration. Substart timepoints to  get duraTion. To cast it to proper unit
				// use duration cast method.
				auto duration_pr = duration_cast<milliseconds>(stop_pr - start_pr);

				// Display results
				std::cout << width << result_delimiter
					<< sd << result_delimiter
					<< thresh << result_delimiter
					<< optimum_params.tx << result_delimiter
					<< optimum_params.ty << result_delimiter
					<< optimum_params.ssq << result_delimiter
					<< optimum_params.num_iter << result_delimiter
					<< duration_pr.count() << "\n";

			}
		}
	}

}


