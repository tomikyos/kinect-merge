Merging Kinect depth maps into a point cloud
============================================

Summary
-------

The program takes a sequence of registered frames captured with the Kinect depth
camera and outputs a non-redundant point cloud. The algorithm uses covariance
matrices to describe the directional variance of the measurements. It does not
limit the captured area and it retains all captured detail. It is partly based
on a paper by Merrell et al. [1].

Process
-------

The input to the algorithm is a sequence of *views* which consist of a depth
image, a color image and extrinsic parameters of the camera. The output is
a point cloud in the ASCII PLY file format.

The views are processed sequentially in a single pass. For each view, the
process consists of two steps: outlier rejection followed by point cloud
refinement. The view being processed is called the *current view*.  Outlier
rejection utilizes two views before and after the current view. They are called
the *adjacent views*. Because of this, the first and last views in the sequence
are not processed.  They do not have two views before and after them to be used
for outlier rejection. Also, outlier rejection limits the minimum number of
views to be 5.  The views which see the same part of the scene as the current
view are called *connected views*.

The Kinect camera produces outliers at depth discontinuities. The outliers are
detected and rejected to improve the quality of the resulting point cloud.
Outlier rejection is based on calculating the *stability* of each point in the
current view. This is done by counting *occlusions* and *free-space violations*.
Occlusions are counted by projecting points from adjacent views into the current
view.  If a point projects in front of a point in the current view, it is an
occlusion. Free-space violations are counted by projecting points from the
current view to all adjacent views. If a point projects in front a point in the
reference view, it is a free-space violation. The stability is defined as the
number of free-space violations substracted from the number of occlusions. If
a point has negative stability, it is considered an outlier and is rejected.

If a projected point is similar enough to a point it overlaps, it does not
contribute to the stability calculation. The similarity metric used is the
Mahalanobis distance between two points, and the threshold for considering two
points to be similar enough is 3 standard deviations.

After outliers have been rejected, existing points in the cloud are refined
using the new points from the current view. This is done by projecting points
from previously processed connected views into the current view. If a projected
point is similar enough to a new point in the current view, the new point is
used to refine the estimate for the existing point taking into account the
directional variances of the points.  The similarity is determined as described
previously. Points which are not used to refine existing points are added to the
resulting point cloud as they are.

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
the `CMAKE_BUILD_TYPE` CMake variable (e.g. `cmake -D CMAKE_BUILD_TYPE=Debug` ../src).

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

The connectivity matrix is an NxN symmetric logical matrix in the OpenCV YAML
format. In this matrix, the value in row Y, column X indicates whether the
corresponding views see the same portion of the scene. If no connectivity matrix
is supplied, a matrix full of ones is used (i.e. all views are considered to be
connected to each other).

### Example

    $ build/kinect_merge dataset-office office.ply

References
----------

[1] Real-time visibility-based fusion of depth maps (P Merrell, A Akbarzadeh, L Wang..., 2007)
