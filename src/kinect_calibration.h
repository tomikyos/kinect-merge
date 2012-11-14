//////////////////////////////////////////////////////////////////
// Kinect capture for the kinect calibration toolbox
// by Daniel Herrera C.
//////////////////////////////////////////////////////////////////
#pragma once

#include <opencv2/opencv.hpp>
#if 0
#include <libfreenect-registration.h>
#endif

namespace kinect_capture {

//////////////////////////////////////////////////////////////////
// CKinectCalibration: encapsulates the calibration parameters of 
//   the kinect and performs conversion between image coordinates
//   and world coordinates.
//////////////////////////////////////////////////////////////////
class CKinectCalibration {
public:
	int rgb_width;
	int rgb_height;
	static const int depth_width=640;
	static const int depth_height=480;

    //disparity2point: converts a point in the depth image to a 3D point
    //in color camera coordinates.
	virtual void disparity2point(int u,int v,short disp,cv::Matx31f &xc)=0;

	//point2rgb: projects a 3D point in color camera coordinates onto the
    //image plane of the color camera and returns the pixel coordinates.
	virtual void point2rgb(const cv::Matx31f &xc, cv::Matx21f &pc)=0;

	virtual void compute_rgb_depthmap(const cv::Mat1s &disp, cv::Mat1f &depth);

	void distort(const float xk, const float yk, const float kc[5], float &xd, float &yd);

protected:
	CKinectCalibration() 
	{}

    CKinectCalibration(int _rgb_width,int _rgb_height):
		rgb_width(_rgb_width),
		rgb_height(_rgb_height)
	{}
};

#if 0
class CMsCalibration: public CKinectCalibration {
public:
	freenect_registration reg;

	virtual void disparity2point(int u,int v,short disp,cv::Matx31f &xc);
	virtual void point2rgb(const cv::Matx31f &xc, cv::Matx21f &pc);
};
#endif

class CBurrusCalibration: public CKinectCalibration {
public:
	CBurrusCalibration():
		CKinectCalibration(640,480),
		calib_fx_d(586.16f), //These constants come from calibration,
        calib_fy_d(582.73f), //replace with your own
        calib_px_d(322.30f),
        calib_py_d(230.07),
        calib_dc1(-0.002851),
        calib_dc2(1093.57),
        calib_fx_rgb(530.11f),
        calib_fy_rgb(526.85f),
        calib_px_rgb(311.23f),
        calib_py_rgb(256.89f),
        calib_R( 0.99999f,   -0.0021409f,     0.004993f,
                    0.0022251f,      0.99985f,    -0.016911f,
                    -0.0049561f,     0.016922f,      0.99984f),
        calib_T(  -0.025985f,   0.00073534f,    -0.003411f)
	{}

    virtual void disparity2point(int u,int v,short disp,cv::Matx31f &xc) {
        cv::Matx31f pd;

        pd(2) = 1.0f / (calib_dc1*(disp - calib_dc2));
        pd(0) = ((u-calib_px_d) / calib_fx_d) * pd(2);
        pd(1) = ((v-calib_py_d) / calib_fy_d) * pd(2);

        xc = calib_R*pd+calib_T;
    }

    virtual void point2rgb(const cv::Matx31f &xc, cv::Matx21f &pc)    {
        pc(0) = xc(0)*calib_fx_rgb/xc(2) + calib_px_rgb;
        pc(1) = xc(1)*calib_fy_rgb/xc(2) + calib_py_rgb;
    }

private:
    const float calib_fx_d;
    const float calib_fy_d;
    const float calib_px_d;
    const float calib_py_d;
    const float calib_dc1;
    const float calib_dc2;
    const float calib_fx_rgb;
    const float calib_fy_rgb;
    const float calib_px_rgb;
    const float calib_py_rgb;
    cv::Matx33f calib_R;
    cv::Matx31f calib_T;
};

class CHerreraCalibration:public CKinectCalibration {
public:
	cv::Matx33f rK;
	float rkc[5];
	float color_error_var;

	cv::Matx33f dK;
	float dkc[5];
	float disp_error_var;

	cv::Matx33f dR;
	cv::Matx31f dT;

	float dc[2];
	float dc_alpha[2];

	cv::Mat1f dc_beta;

	CHerreraCalibration() {}
	void load(const std::string &filename);
	float undistort_disp(int u, int v, float disp);
    virtual void disparity2point(int u,int v,short disp,cv::Matx31f &xc);
    virtual void point2rgb(const cv::Matx31f &xc, cv::Matx21f &pc);
};

}
