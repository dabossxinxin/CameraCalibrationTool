#include "CommonFunctions.h"
#include "LineScanCalibration.h"

bool LineScanCalibration::Update() 
{
	// 判断参数是否正确输入
	if (mObjectPoints.empty() || mImagePoints.empty()) {
		std::cerr << "Object Points || Image Points Size Empty..." << std::endl;
		return false;
	}
	if (mObjectPoints.size() != mImagePoints.size()) {
		std::cerr << "Object Points's size not equal to Image Points's..." << std::endl;
		return false;
	}

	if (!this->InitialEstimate()) {
		std::cerr << "Initial Estimate Failed..." << std::endl;
		return false;
	}
	if (!this->OptimizeEstimate()) {
		std::cerr << "Optimize Estimate Failed..." << std::endl;
		return false;
	}

	return true;
}

bool LineScanCalibration::InitialEstimate() 
{
	CommonFunctions::ConditionPrint("Start Initial Estimate...");

	CommonFunctions::ConditionPrint("End Initial Estimate");
	return true;
}

bool LineScanCalibration::OptimizeEstimate() 
{
	CommonFunctions::ConditionPrint("Start Optimize Estimate...");

	CommonFunctions::ConditionPrint("End Optimize Estimate");
	return true;
}