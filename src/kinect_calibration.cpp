#include "kinect_calibration.h"

namespace kinect_capture {

void CKinectCalibration::compute_rgb_depthmap(const cv::Mat1s &disp, cv::Mat1f &depth) {
	const int SPLAT_RADIUS = 0;

	depth = cv::Mat1f::zeros(rgb_height,rgb_width);

    for(int v=0; v<disp.rows; v++)
        for(int u=0; u<disp.cols; u++) {
            const short d = disp(v,u);
            if(d==2047) 
                continue;

            cv::Matx31f xc;
            cv::Matx21f p;
            int uci,vci;
            int uci0,vci0;
			float z_new;

            disparity2point(u,v,d,xc);
            point2rgb(xc,p);
            uci0 = (int)(p(0)+0.5f);
            vci0 = (int)(p(1)+0.5f);
			z_new = xc(2);

			for(uci=uci0-SPLAT_RADIUS; uci<=uci0+SPLAT_RADIUS; uci++)
				for(vci=vci0-SPLAT_RADIUS; vci<=vci0+SPLAT_RADIUS; vci++) {
					if(uci<0 || uci>=depth.cols || vci<0 || vci>=depth.rows)
						continue;
            
					float &z_old = depth(vci,uci);
					if(z_old == 0 || z_old > z_new)
						z_old = z_new;
				}
        }
}

void CKinectCalibration::distort(const float xk, const float yk, const float kc[5], float &xd, float &yd) {
	float r2 = xk*xk + yk*yk;
	float r4 = r2*r2;
	float r6 = r4*r2;
	float rc = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
	float xy = xk*yk;
	float dx = 2*kc[2]*xy + kc[3]*(r2+2*xk*xk);
	float dy = 2*kc[3]*xy + kc[2]*(r2+2*yk*yk);

	xd = rc*xk + dx;
	yd = rc*yk + dy;
}

#if 0
void CMsCalibration::disparity2point(int u,int v,short disp,cv::Matx31f &xc) {
	const float DEPTH_X_RES = 640;
	const float DEPTH_Y_RES = 480;
	uint16_t wz = reg.raw_to_mm_shift[disp];
	
	float ref_pix_size = reg.zero_plane_info.reference_pixel_size;
	float ref_distance = reg.zero_plane_info.reference_distance;
	float xfactor = 2*ref_pix_size * wz / ref_distance;
	float yfactor = (1024/480)*ref_pix_size * wz / ref_distance;
	//xfactor = ref_pix_size * wz / ref_distance;
	//float yfactor = xfactor;
	xc(0) = (u - DEPTH_X_RES/2) * xfactor / 1000;
	xc(1) = (v - DEPTH_Y_RES/2) * yfactor / 1000;
	xc(2) = (float)(wz / 1000);
}
#endif

void point2rgb(const cv::Matx31f &xc, cv::Matx21f &pc) {
}

void CHerreraCalibration::load(const std::string &filename) {
	const int COLOR_CAMERA_IDX=0;
	cv::FileStorage fs;
	std::stringstream s;
	cv::Mat m;

	fs.open(filename,cv::FileStorage::READ);

	s << "rsize" << (COLOR_CAMERA_IDX+1);
	fs[s.str().c_str()] >> m;
	rgb_height = m.at<float>(0);
	rgb_width = m.at<float>(1);

	s.str("");
	s << "rK" << (COLOR_CAMERA_IDX+1);
	fs[s.str().c_str()] >> m;
	rK = m;

	s.str("");
	s << "rkc" << (COLOR_CAMERA_IDX+1);
	fs[s.str().c_str()] >> m;
	memcpy(rkc,m.data,sizeof(rkc));

	fs["color_error_var"] >> m;
	color_error_var = m.at<float>(0,COLOR_CAMERA_IDX);

	fs["dK"] >> m;
	dK = m;

	fs["dkc"] >> m;
	memcpy(dkc,m.data,sizeof(dkc));

	fs["dR"] >> m;
	dR = m;

	fs["dt"] >> m;
	dT = m;

	fs["dc"] >> m;
	memcpy(dc,m.data,sizeof(dc));

	fs["dc_alpha"] >> m;
	memcpy(dc_alpha,m.data,sizeof(dc));

	fs["dc_beta"] >> m;
	dc_beta = m;

	fs["depth_error_var"] >> m;
	disp_error_var = m.at<float>(0);
}

float CHerreraCalibration::undistort_disp(int u, int v, float disp) {
	return disp + dc_beta(v,u)*std::exp(dc_alpha[0] - dc_alpha[1]*disp);
}

void CHerreraCalibration::disparity2point(int u,int v,short disp,cv::Matx31f &xc) {
	float disp_k = undistort_disp(u,v,disp);
	float z = 1.0f / (dc[1]*disp_k + dc[0]);
	float xk = (u-dK(0,2))/dK(0,0);
	float yk = (v-dK(1,2))/dK(1,1);
	float xd,yd;
	cv::Matx31f x;

	distort(xk,yk,dkc,xd,yd);
	x(0) = xd*z;
	x(1) = yd*z;
	x(2) = z;

	xc = dR*x + dT;
}

void CHerreraCalibration::point2rgb(const cv::Matx31f &xc, cv::Matx21f &pc) {
	float xk = xc(0)/xc(2);
	float yk = xc(1)/xc(2);
	float xd,yd;
	distort(xk,yk,rkc,xd,yd);
	pc(0) = rK(0,0)*xd+rK(0,2);
	pc(1) = rK(1,1)*yd+rK(1,2);
}

}
