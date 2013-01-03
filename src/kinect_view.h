#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <boost/multi_array.hpp>
#include <boost/shared_ptr.hpp>

#include "kinect_calibration.h"

namespace kinect_merge {

using kinect_capture::CKinectCalibration;

static const int IMAGE_WIDTH = 640;
static const int IMAGE_HEIGHT = 480;
static const int ADJACENCY_SIZE = 2;

struct CPoint {
    typedef boost::shared_ptr<CPoint> ptr;
    typedef boost::shared_ptr<const CPoint> const_ptr;

    CPoint(const cv::Matx31f& position, const cv::Matx33f& covariance, const cv::Matx<unsigned char, 3, 1>& color);

    cv::Matx31f pos; // Position
    cv::Matx33f cov; // Covariance matrix
    cv::Matx<unsigned int, 3, 1> col_sum; // Color sum
    unsigned int num_merged; // Number of points included in the color sum.

    cv::Matx<unsigned char, 3, 1> get_color() const;
};

class CView {
public:
    CView(const std::string &exstrinsic_file,
          const std::string &disparity_file,
          const std::string &color_file,
          CKinectCalibration& calibration);

    // Does the original depthmap have a measurement at the given pixel?
    bool has_measurement(int u, int v) const { return original_pointmap[v][u]; }

    // Has a point been added to the point cloud from the given pixel?
    bool has_point(int u, int v) const { return pointmap[v][u]; }

    // Return the depth of the original measurement at the given pixel.
    float get_depth(int u, int v) const { assert(has_measurement(u, v)); return depthmap(v, u); }

    // Return a point corresponding to a pixel in the original depth map.
    const CPoint& get_original_point(int u, int v) const { assert(has_measurement(u, v)); return *original_pointmap[v][u]; }

    // Return a point corresponding to a pixel. The point might have been refined by later measurements.
    CPoint& get_point(int u, int v) { assert(has_point(u, v)); return *pointmap[v][u]; }

    // Insert all original points into the cloud unmodified.
    void insert_into_point_cloud(std::vector<CPoint::const_ptr> &point_cloud);

    // Project a global point onto the image plane of this view.
    void project(const CPoint &global_point, int &proj_u, int &proj_v, float &proj_depth) const;

    // Add the points of this view to the point cloud.
    void merge(boost::ptr_vector<CView> views,
               unsigned int view_idx,
               const cv::Mat view_connectivity,
               std::vector<CPoint::ptr> &global_point_cloud);

    // Statistics
    int get_num_missing() const { return num_missing; }
    int get_num_outliers() const { return num_outliers; }
    int get_num_added() const { return num_added; }

private:
    // Project a local point onto the image plane of this view.
    void to_image_plane(const cv::Matx41f &local_pos, int &uc, int &vc) const;

    // Detect and mark outliers as rejected.
    void reject_outliers(boost::ptr_vector<CView> views,
                         unsigned int view_idx,
                         std::vector<std::vector<bool> > &measurement_accepted);

    // Update existing points in the connected views using new measurements that are similar enough.
    void refine_points(CView &connected_view,
                       const std::vector<std::vector<bool> > &measurement_accepted,
                       std::vector<std::vector<bool> > &measurement_used) const;

    CKinectCalibration& calibration;

    cv::Matx44f inv_transformation;
    cv::Matx<float, IMAGE_HEIGHT, IMAGE_WIDTH> depthmap;

    // Contains the global points backprojected from the original depthmap.
    // Has dimensions IMAGE_HEIGHTxIMAGE_WIDTH.
    boost::multi_array<CPoint::const_ptr, 2> original_pointmap;

    // Contains the global points added to the point cloud from this view.
    // These might have been refined by later measurements.
    // Has dimensions IMAGE_HEIGHTxIMAGE_WIDTH.
    boost::multi_array<CPoint::ptr, 2> pointmap;

    // Statistics
    int num_missing;
    int num_outliers;
    int num_added;

#ifdef DEBUG
public:
    // The measurement acceptance is used for outputting debug images.
    std::vector<std::vector<bool> > measurement_accepted;
#endif
};

}
