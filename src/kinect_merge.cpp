#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <boost/filesystem.hpp>
#include <boost/timer.hpp>
#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/program_options.hpp>
#include <boost/make_shared.hpp>
#include <opencv2/opencv.hpp>

#include "kinect_view.h"
#include "kinect_calibration.h"

using namespace kinect_merge;
using kinect_capture::CHerreraCalibration;
namespace po = boost::program_options;

#ifdef DEBUG
void print_matrix(const cv::Mat &m) {
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            std::cout << m.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}
#endif

static void print_usage(std::string program_name, po::options_description desc) {
    std::cerr << "Usage: " << program_name << " [options] capture_dir output_file" << std::endl;
    std::cerr << desc << std::endl;
}

static void parse_options(int argc, const char **argv, po::variables_map &vm) {
    po::options_description desc;
    desc.add_options()
        ("help,h",              "display this help and exit")
        ("covscalexy,sxy",      boost::program_options::value<float>()->default_value(DEFAULT_COVARIANCE_SCALE_XY),
                                "scale factor for measurement x- and y-variances")
        ("covscalez,sz",        boost::program_options::value<float>()->default_value(DEFAULT_COVARIANCE_SCALE_Z),
                                "scale factor for measurement z-variance")
        ("connectivity,c",      boost::program_options::value<std::string>(),
                                "OpenCV YAML file containing the view connectivity matrix C")
        ("outlierrejection,o",  "enable outlier rejection")
        ("debug,d",             "output debug files")
        ("capture_dir",         "directory to read the capture data from")
        ("output_file",         "PLY file to write the resulting point cloud to");

    po::positional_options_description pd;
    pd.add("capture_dir", 1);
    pd.add("output_file", 1);

    po::parsed_options parsed = po::basic_command_line_parser<char>(argc, argv).options(desc)
                                                                               .allow_unregistered()
                                                                               .positional(pd)
                                                                               .run();
    po::store(parsed, vm);
    po::notify(vm);

    if(vm.count("help") || !vm.count("capture_dir") || !vm.count("output_file")) {
        print_usage(argv[0], desc);
        exit(0);
    }
}

// Load all disparity, extrinsic and color image files for views from the given directory.
//
// The files must be named 0000-d.pgm, 0000-p.yml, 0000-r.jpg, 0001-d.pgm etc.
// The color image file can alternatively be named 0000-c1.ppm.
static void load_views(const std::string &capture_dir,
                       boost::ptr_vector<CView> &views,
                       float covariance_scale_xy,
                       float covariance_scale_z,
                       CKinectCalibration &calibration) {
    int view_idx = 0;
    while(true) {
        std::ostringstream view_file_base_stream;
        view_file_base_stream << capture_dir;
        view_file_base_stream << "/";
        view_file_base_stream << std::setfill('0') << std::setw(4) << view_idx;
        std::string view_file_base = view_file_base_stream.str();

        // The color file can be in JPG or PPM format.

        std::string disparity_file = view_file_base + "-d.pgm";
        std::string extrinsic_file = view_file_base + "-p.yml";
        std::string jpg_color_file = view_file_base + "-r.jpg";
        std::string ppm_color_file = view_file_base + "-c1.ppm";

        std::string color_file;
        if(boost::filesystem::exists(jpg_color_file)) {
            color_file = jpg_color_file;
        } else if(boost::filesystem::exists(ppm_color_file)) {
            color_file = ppm_color_file;
        }

        if(!boost::filesystem::exists(disparity_file) ||
           !boost::filesystem::exists(extrinsic_file) ||
           color_file.empty()) {
            break;
        }

        views.push_back(new CView(extrinsic_file,
                                  disparity_file,
                                  color_file,
                                  covariance_scale_xy,
                                  covariance_scale_z,
                                  calibration));
        std::cout << "." << std::flush; // Progress output
        view_idx++;
    }
}

// Output a percentage indicating the progress of a process.
static void print_progress(int current_index, int num_indices) {
    std::cout << (int)((float)current_index / num_indices * 100) << " %" << std::flush << "\r";
}

// Write a point cloud from a vector into a file in the PLY ASCII format.
template<typename point_ptr>
static void output_point_cloud(const std::vector<point_ptr> &point_cloud, const std::string &filename) {
    std::cout << "Writing point cloud to " << filename << std::endl;

    std::ofstream ply_file(filename.c_str());

    if(!ply_file) {
        std::cerr << "Error opening output file" << std::endl;
        exit(1);
    }

    ply_file << "ply" << std::endl;
    ply_file << "format ascii 1.0" << std::endl;
    ply_file << "element vertex " << point_cloud.size() << std::endl;
    ply_file << "property float x" << std::endl;
    ply_file << "property float y" << std::endl;
    ply_file << "property float z" << std::endl;
    ply_file << "property uchar red" << std::endl;
    ply_file << "property uchar green" << std::endl;
    ply_file << "property uchar blue" << std::endl;
    ply_file << "end_header" << std::endl;
    for(unsigned int i = 0; i < point_cloud.size(); i++) {
        if((i & 0xffff) == 0) {
            print_progress(i, point_cloud.size());
        }

        point_ptr p = point_cloud[i];
        cv::Matx<unsigned char, 3, 1> color = p->get_color();
        ply_file << p->pos(0) << " " << p->pos(1) << " " << p->pos(2) << " ";
        ply_file << static_cast<int>(color(2)) << " "
                 << static_cast<int>(color(1)) << " "
                 << static_cast<int>(color(0)) << std::endl;

        if(!ply_file) {
            std::cerr << "Error writing point cloud" << std::endl;
            exit(2);
        }
    }
    ply_file.close();
}

// Statistics functions

template<typename T>
static float sum(const std::vector<T> &values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
}

template<typename T>
static float average(const std::vector<T> &values) {
    return sum(values) / values.size();
}

template<typename T>
static float minimum(const std::vector<T> &values) {
    return *std::min_element(values.begin(), values.end());
}

template<typename T>
static float maximum(const std::vector<T> &values) {
    return *std::max_element(values.begin(), values.end());
}

int main(int argc, const char **argv) {
    po::variables_map vm;
    parse_options(argc, argv, vm);

    bool debug = vm.count("debug");
    std::string capture_dir = vm["capture_dir"].as<std::string>();
    std::string output_file = vm["output_file"].as<std::string>();
    float covariance_scale_xy = vm["covscalexy"].as<float>();
    float covariance_scale_z = vm["covscalez"].as<float>();
    bool outlier_rejection = vm.count("outlierrejection");

    // Statistics variables
    float time_loading_stat;
    std::vector<float> time_merging_stats;
    std::vector<float> num_missing_stats;
    std::vector<float> num_outliers_stats;
    std::vector<float> num_added_stats;

    if(!boost::filesystem::exists("calib.yml")) {
        std::cerr << "calib.yml does not exist" << std::endl;
        exit(9);
    }

    CHerreraCalibration calibration;
    calibration.load("calib.yml");

    boost::ptr_vector<CView> views;
    std::cout << "Loading views" << std::flush;
    boost::timer timer;
    load_views(capture_dir, views, covariance_scale_xy, covariance_scale_z, calibration);
    time_loading_stat = timer.elapsed();
    std::cout << views.size() << " views" << std::endl;

    if(outlier_rejection) {
        // Each view must have two views on each side for the outlier rejection step.
        if(views.size() < 2 * ADJACENCY_SIZE + 1) {
            std::cerr << "Need at least " << 2 * ADJACENCY_SIZE + 1 << " views" << std::endl;
            exit(3);
        }
    }

    if(debug) {
        // Output all the points overlayed in one cloud.
        std::vector<CPoint::const_ptr> overlayed_point_cloud;
        std::cout << "Overlaying point clouds" << std::endl;
        for(unsigned int i = ADJACENCY_SIZE; i < views.size() - ADJACENCY_SIZE; i++) {
            print_progress(i, views.size());
            views[i].insert_into_point_cloud(overlayed_point_cloud);
        }
        output_point_cloud<CPoint::const_ptr>(overlayed_point_cloud, "overlayed.ply");
    }

    // Get a view connectivity matrix.
    cv::Mat view_connectivity;
    if(vm.count("connectivity")) {
        std::string connectivity_file = vm["connectivity"].as<std::string>();

        if(!boost::filesystem::exists(connectivity_file)) {
            std::cerr << "Connectivity file " << connectivity_file << " does not exist" << std::endl;
            exit(8);
        }

        try {
            cv::FileStorage fs(connectivity_file, cv::FileStorage::READ);
            cv::Mat temp;
            fs["C"] >> temp;
            temp.convertTo(view_connectivity, cv::DataType<bool>::type);
        } catch(cv::Exception &e) {
            std::cerr << "Error reading connectivity matrix: " << e.what() << std::endl;
            exit(7);
        }
    } else {
        view_connectivity = cv::Mat_<bool>::ones(views.size(), views.size());
    }

    if(static_cast<unsigned int>(view_connectivity.rows) < views.size() ||
       static_cast<unsigned int>(view_connectivity.cols) < views.size()) {
        std::cerr << "Connectivity matrix has dimensions " << view_connectivity.cols << "x" << view_connectivity.rows
                  << " instead of " << views.size() << "x" << views.size() << std::endl;
        exit(8);
    }

    unsigned int first_view = outlier_rejection ? ADJACENCY_SIZE : 0;
    unsigned int last_view = outlier_rejection ? views.size() - ADJACENCY_SIZE : views.size();

    std::cout << "Merging views" << std::endl;
    std::vector<CPoint::ptr> global_point_cloud;
    for(unsigned int view_idx = first_view; view_idx < last_view; view_idx++) {
        if(outlier_rejection) {
            print_progress(view_idx - ADJACENCY_SIZE, views.size() - 2 * ADJACENCY_SIZE);
        } else {
            print_progress(view_idx, views.size());
        }

        timer.restart();
        CView &view = views[view_idx];
        view.merge(views,
                   view_idx,
                   view_connectivity,
                   outlier_rejection,
                   global_point_cloud);

        // Collect statistics.
        float num_pixels = IMAGE_WIDTH * IMAGE_HEIGHT;
        time_merging_stats.push_back(timer.elapsed());
        num_missing_stats.push_back(view.get_num_missing() / num_pixels * 100);
        if(outlier_rejection) {
            num_outliers_stats.push_back(view.get_num_outliers() / (num_pixels - view.get_num_missing()) * 100);
        }
        num_added_stats.push_back(view.get_num_added() / (num_pixels - view.get_num_missing() - view.get_num_outliers()) * 100);

#ifdef DEBUG
        if(debug) {
            // Output images showing the computed depth values.

            cv::Matx<float, IMAGE_HEIGHT, IMAGE_WIDTH> depthmap;
            float max_depth = 0;
            for(int v = 0; v < IMAGE_HEIGHT; v++) {
                for(int u = 0; u < IMAGE_WIDTH; u++) {
                    float depth = view.has_measurement(u, v) && view.measurement_accepted[v][u]
                                ? view.get_depth(u, v)
                                : 0;
                    if(depth > max_depth) {
                        max_depth = depth;
                    }
                    depthmap(v, u) = depth;
                }
            }

            depthmap = depthmap * (1 / max_depth) * 255;

            std::ostringstream depthmap_filename;
            depthmap_filename << "depthmap-" << view_idx << ".png";
            cv::imwrite(depthmap_filename.str(), depthmap);

            // Output images showing the state of each point.

            cv::Mat state(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
            for(int v = 0; v < IMAGE_HEIGHT; v++) {
                for(int u = 0; u < IMAGE_WIDTH; u++) {
                    cv::Vec3b color; // in BGR format
                    if(view.has_measurement(u, v)) {
                        if(view.measurement_used[v][u]) {
                            // The point was used to refine an existing point.
                            color[0] = 128;
                            color[1] = 255;
                            color[2] = 128;
                        } else if(view.measurement_accepted[v][u]) {
                            // The point was added to the cloud.
                            color[0] = 255;
                            color[1] = 128;
                            color[2] = 128;
                        } else {
                            // The point was an outlier.
                            color[0] = 128;
                            color[1] = 128;
                            color[2] = 255;
                        }
                    } else {
                        // There was no measurement for the pixel.
                        color[0] = 213;
                        color[1] = 213;
                        color[2] = 213;
                    }
                    state.at<cv::Vec3b>(v, u) = color;
                }
            }

            std::ostringstream state_filename;
            state_filename<< "state-" << view_idx << ".png";
            cv::imwrite(state_filename.str(), state);
        }
#endif
    }
    output_point_cloud<CPoint::ptr>(global_point_cloud, output_file);

    // Output statistics.
    unsigned int num_original_points = views.size() * IMAGE_WIDTH * IMAGE_HEIGHT - sum(num_missing_stats);
    unsigned int num_resulting_points = global_point_cloud.size();
    std::cout << "Statistics:" << std::endl;
    std::cout << "Loading views: avg " << time_loading_stat / views.size() << " s" << std::endl;
    std::cout << "Merging views: avg " << average(time_merging_stats) << " s, min " << minimum(time_merging_stats) << " s, max " << maximum(time_merging_stats) << " s" << std::endl;
    std::cout << "Missing measurements: avg " << average(num_missing_stats) << " %, min " << minimum(num_missing_stats) << " %, max " << maximum(num_missing_stats) << " %" << std::endl;
    if(outlier_rejection) {
        std::cout << "Rejected measurements: avg " << average(num_outliers_stats) << " %, min " << minimum(num_outliers_stats) << " %, max " << maximum(num_outliers_stats) << " %" << std::endl;
    }
    std::cout << "Added points: avg " << average(num_added_stats) << " %, min " << minimum(num_added_stats) << " %, max " << maximum(num_added_stats) << " %" << std::endl;
    std::cout << "Points in the original cloud: " << num_original_points << std::endl;
    std::cout << "Points in the resulting cloud: " << num_resulting_points << std::endl;
    std::cout << "Ratio: " << (float)num_resulting_points / num_original_points * 100 << " %" << std::endl;

    return 0;
}
