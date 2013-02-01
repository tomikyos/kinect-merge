Merging Kinect depth maps into a nonredundant point cloud
============================================

Summary
-------

The program takes a sequence of registered views captured with the Kinect depth
camera and outputs a nonredundant point cloud. The algorithm creates the point
cloud incrementally and uses overlapping measurements to reduce the directional
variance of estimated point positions. It does not limit the captured volume, it
allows for varying levels of detail, and it doesn't require any parameters from
the user.

Process
-------

The input to the algorithm is a sequence of views which consist of a depth
map, a color image and extrinsic parameters of the camera. In addition,
connectivity information for the views can be supplied to increase
performance.  The output is a point cloud in the ASCII PLY file format.

The views are processed sequentially in a single pass. When a new view is being
added, existing points from the cloud are projected onto the pixel grid of the
new view. If a new measurement resides near existing points, a new estimate is
created for each existing point with the best linear unbiased estimator (BLUE).
Then, the Mahalanobis distances from the new estimate to both the existing point
and new measurement is calculated. If both distances are smaller than a threshold,
the existing point in the cloud is replaced with the new estimate. This results
in reduced variance for the point with no added redundancy in the cloud. If the
distance condition is not met for any new estimate (i.e. the new measurement
wasn't used to refine any existing point), the measurement represents a novel
point of surface and is added to the cloud as a new point.

Prerequisites
-------------

* CMake 2.8
* OpenCV 2.3.1
* Boost 1.49.0 (the system, filesystem and program_options libraries)

Building
--------

The program has been developed and tested on Linux.

By default, the root of an OpenCV build is expected to be found at
`../../OpenCV-2.3.1/build` relative to the `src` directory. This can be changed
using the `OpenCV_DIR` CMake variable.

The steps for building are:

    $ mkdir build
    $ cd build
    $ cmake ../src
    $ make

For additional debug file output, a debug build can be performed by specifying
the `CMAKE_BUILD_TYPE` CMake variable (e.g. `cmake -D CMAKE_BUILD_TYPE=Debug ../src`).

Usage
-----

The input to the program is a sequence of pgm, yml and jpg files, one of each
per view.  The files should be named sequentially in the format shown in the
parenthesis.

    pgm:    The grey scale disparity image file (e.g. 0000-d.pgm).
    yml:    The extrinsic camera parameter data file (e.g. 0000-p.yml).
    jpg:    The color image file (e.g. 0000-r.jpg or 0000-c1.jpg).

The extrinsic parameters consist of the 3x3 camera rotation matrix R and
the 3x1 camera translation matrix T in the OpenCV YAML format.

The program requires calibration data to be found in a `calib.yml` file in the
current directory.

### Command-line

    ./kinect_merge [options] capture_dir output_file

        -c [ --connectivity ] FILE      The camera connectivity matrix.
        -d [ --debug ]                  Output additional debug files.
        --covscalexy FACTOR             Scale factor for measurement x- and y-variances (default: 40).
        --covscalez FACTOR              Scale factor for measurement z-variance (default: 20).

The connectivity matrix is an NxN symmetric logical matrix in the OpenCV YAML
format. In this matrix, the value in row Y, column X indicates whether the
corresponding views see the same portion of the scene. If no connectivity matrix
is supplied, a matrix full of ones is used (i.e. all views are considered to be
connected to each other).

The covariance scaling options are supplied for accounting for error resulting
from view alignment.  The default values are suitable for the alignment process
used for included data set. The scale factors should not be data set specific.

### Example

    $ build/kinect_merge dataset-office office.ply
