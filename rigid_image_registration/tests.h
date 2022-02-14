/*
Header file for function tests

Author:
Nemanja Sakac, Student
Faculty of Technical Sciences, University of Novi Sad
Novi Sad, Serbia
Date: 2021.
*/

#pragma once

#include "rig_reg_tools.h"

void test_znorm(const cv::Mat& img);
void test_interp(cv::Mat& img);
void test_basic_func(std::string& img_ref_name, std::string& img_reg_name);
void test_convergence_sgd(std::string& img_ref_name, std::string& img_reg_name);
void test_convergence_pr(std::string& img_ref_name, std::string& img_reg_name);
