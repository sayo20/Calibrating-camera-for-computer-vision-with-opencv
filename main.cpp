#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

using namespace cv;
using namespace std;

const int square_length = 2.49;
const Size board_dimensin = Size(9, 6);


static void saveCameraParam(String filename, Mat& camera_matrix, Mat& distor_coeff, const
    vector<Mat>& r_vec, const vector<Mat>& t_vec) {
    /*
     This is a helper function to create an xml file with camera matrix, distortion matrix, rotation and translation vector.
     */
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "distort_coeff" << distor_coeff;
    fs << "r_vec" << r_vec;
    fs << "t_vec" << t_vec;

}


void createKnownBoardPosition(Size board_size, float sqr_length, vector<Point3f>& corners) {
    /*
     We get the know square positions on the board
     board_size: is the dimension of the board
     sqr_length : length of one side of the square
     corners: stores x, y, z co-ordinates of where expect each square to appear(the intersections between 2 squares) but z here is 0 since we have 2D images
     */

    for (int x = 0; x < board_size.height; x++) {

        for (int y = 0; y < board_size.width; y++) {

            corners.push_back(Point3f(y * sqr_length, x * sqr_length, 0));

        }
    }
}


void cameraCalibrate() {

    /*
     Here we calibrate our camera by reading the stored path names to our images from a text file.
     "realworld_corner_points" :holds the x,y,z co-ordinates values of the images in real world
     Points_founds: holds corners returned by findChessboardCorners method
     get_corners: save all the corners for every image image used for callibration
     r_vec and t_vec: holds the rotation and translation vector returned by calibrateCamera mrthod
     */
    vector<Mat> imgs;
    vector<Mat> tmp_img_holder;
    vector<string> tmp;
    vector<Point2f> points_found;
    vector<vector<Point3f>> realworld_corner_points(1);
    vector<vector<Point2f>> get_corners;
    vector<Mat> r_vec, t_vec;
    Mat camera_matrix = Mat::eye(3, 3, CV_64F);
    Mat distortion_coeff = Mat::zeros(8, 1, CV_64F);
    string str;
    vector<int> indexes;
    std::ifstream filename("imagesList.txt");

    while (getline(filename, str))
    {
        if (str.size() > 0) {
            Mat tmp = imread(str);
            imgs.push_back(tmp);
        }
    }


    /*
     We manually changed the values of i to see the changes in intrinsic matrix. so we checked callibration when i starts at 0, 1 and 2.
     on first run we do from i= 0 to 35(total img size is 35)
     on second run we do from i= 0 to 29(total img size is 35)
     on third run we do everything but images 22,23, and 24 (total img size is 35)
     */

    for (int i = 0; i < imgs.size(); i++) {
        bool found = findChessboardCorners(imgs[i], board_dimensin, points_found, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        if (found) {
            get_corners.push_back(points_found);
        }
    }

    createKnownBoardPosition(board_dimensin, square_length, realworld_corner_points[0]);
    realworld_corner_points.resize(get_corners.size(), realworld_corner_points[0]);

    calibrateCamera(realworld_corner_points, get_corners, board_dimensin, camera_matrix, distortion_coeff, r_vec, t_vec);
    saveCameraParam("calibrationDetails.xml", camera_matrix, distortion_coeff, r_vec, t_vec);
}
void drawCube(Mat img, vector< Point2f >& imagePoints) {

    /*
     This function draws the line of the cube. We first draw the lower square, then the upper square, then the lines in between
     */
    line(img, imagePoints[0], imagePoints[1], Scalar(0, 0, 255), 3);
    line(img, imagePoints[1], imagePoints[4], Scalar(0, 0, 255), 3);
    line(img, imagePoints[3], imagePoints[4], Scalar(0, 0, 255), 3);
    line(img, imagePoints[2], imagePoints[0], Scalar(0, 0, 255), 3);

    //upper square
    cv::line(img, imagePoints[6], imagePoints[5], Scalar(0, 0, 255), 3);
    cv::line(img, imagePoints[7], imagePoints[2], Scalar(0, 0, 255), 3);
    cv::line(img, imagePoints[7], imagePoints[6], Scalar(0, 0, 255), 3);

    // inbetween
    cv::line(img, imagePoints[4], imagePoints[6], Scalar(0, 0, 255), 3);
    cv::line(img, imagePoints[1], imagePoints[5], Scalar(0, 0, 255), 3);
    cv::line(img, imagePoints[2], imagePoints[5], Scalar(0, 0, 255), 3);
    cv::line(img, imagePoints[3], imagePoints[7], Scalar(0, 0, 255), 3);

}

vector<Point3f> getAxispoints() {
    /*
     This function gets all the 8 axis points in order to draw on the imagepoints.
     */

    vector<Point3f> cube_axisPoints;
    int length = square_length;
    cube_axisPoints.push_back(Point3f(0, 0, 0));
    cube_axisPoints.push_back(Point3f(length, 0, 0));
    cube_axisPoints.push_back(Point3f(0, length, 0));
    cube_axisPoints.push_back(Point3f(0, 0, -length));

    cube_axisPoints.push_back(Point3f(length, 0, -length));
    cube_axisPoints.push_back(Point3f(length, length, 0));
    cube_axisPoints.push_back(Point3f(length, length, -length));
    cube_axisPoints.push_back(Point3f(0, length, -length));

    return cube_axisPoints;

}

void drawCubesWithOfflineImages() {
    /*
     This method deals with offline images. First it takes in all .
     */
    Mat frame, draw_to_frame;
    vector<Mat> imgs;
    String cam_frame_name = "Webcam";
    vector<Point3f> axisPoints;
    vector< Point2f > cube_imagePoints;
    vector<Point3f> objpoint;
    vector< Point2f > imagePoints;
    Point2f testpoint;
    vector<vector<Point2f>> save_corners(1);
    vector<vector<Point3f>> realworld_corner_points(1);
    vector<Point2f> get_corners;
    bool found;
    Mat intrinsics, distortion, rvect;
    Mat rvec, tvec;
    string str;


    axisPoints = getAxispoints();

    createKnownBoardPosition(board_dimensin, square_length, objpoint);

    //create a function for rading out calibration imgs since we're using it twice
    std::ifstream filename("imagesList.txt");

    while (getline(filename, str))
    {
        if (str.size() > 0) {
            Mat tmp = imread(str);
            imgs.push_back(tmp);
        }
    }

    for (int i = 0; i < imgs.size(); i++) {
        found = findChessboardCorners(imgs[i], board_dimensin, get_corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);    if (found) {
            FileStorage fs("calibrationDetails.xml", FileStorage::READ);

            fs["camera_matrix"] >> intrinsics;
            fs["distort_coeff"] >> distortion;

            solvePnP(objpoint, get_corners, intrinsics, distortion, rvec, tvec);
            projectPoints(axisPoints, rvec, tvec, intrinsics, distortion, imagePoints);

            line(imgs[i], imagePoints[0], imagePoints[1], Scalar(255, 0, 0), 3);
            line(imgs[i], imagePoints[0], imagePoints[2], Scalar(0, 255, 0), 3);
            line(imgs[i], imagePoints[0], imagePoints[3], Scalar(0, 0, 255), 3);

            drawCube(imgs[i], imagePoints);

            string file_name = "cubeimgs/imgs" + to_string(i) + ".jpg";
            imwrite(file_name, imgs[i]);
        }
        else {
            cout << "Image " << to_string(i) << " is a bad image";
        }
    }
}


void onlinePhase() {
    /*
     In this Function, we draw the the cube on the chessboard while straming from the web cam
     */
    Mat frame, draw_to_frame;
    vector<Mat> imgs;
    String cam_frame_name = "Webcam";
    int frame_per_sec = 20;
    vector<Point3f> axisPoints;
    vector<Point3f> objpoint;
    vector< Point2f > imagePoints;
    vector<vector<Point2f>> save_corners(1);
    vector<vector<Point3f>> realworld_corner_points(1);
    vector<Point2f> get_corners;
    bool found;
    Mat intrinsics, distortion, rvect;
    Mat rvec, tvec;

    VideoCapture vid(0);

    if (!vid.isOpened()) {
        cout << "camera not opened";
    }

    namedWindow(cam_frame_name, WINDOW_AUTOSIZE);
    axisPoints = getAxispoints();


    createKnownBoardPosition(board_dimensin, square_length, objpoint);
    while (true) {
        if (!vid.read(frame)) {
            break;
        }

        found = findChessboardCorners(frame, board_dimensin, get_corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        frame.copyTo(draw_to_frame);
        drawChessboardCorners(draw_to_frame, board_dimensin, get_corners, found);

        if (found) {
            FileStorage fs("calibrationDetails.xml", FileStorage::READ);

            fs["camera_matrix"] >> intrinsics;
            fs["distort_coeff"] >> distortion;

            solvePnP(objpoint, get_corners, intrinsics, distortion, rvec, tvec);
            projectPoints(axisPoints, rvec, tvec, intrinsics, distortion, imagePoints);

            line(draw_to_frame, imagePoints[0], imagePoints[1], Scalar(255, 0, 0), 3);
            line(draw_to_frame, imagePoints[0], imagePoints[2], Scalar(0, 255, 0), 3);
            line(draw_to_frame, imagePoints[0], imagePoints[3], Scalar(0, 0, 255), 3);
            drawCube(draw_to_frame, imagePoints);
            imshow(cam_frame_name, draw_to_frame);
        }
        else {
            imshow(cam_frame_name, frame);
        }
        char c = waitKey(1000 / frame_per_sec);

        if (c == 27) {
            imwrite("intrinsics1.jpg", draw_to_frame);
            break;
        }
    }
}

int main(int argc, const char* argv[]) {
    /*If the code does not run - specify the specific files/image paths in the filestorage functions and imagelist files/

    /*Run for online drawing of the cubes*/
    //onlinePhase();

    /*Run for calibartion*/
    //cameraCalibrate();

    /*Run for offline phase*/
    //drawCubesWithOfflineImages();


    return 0;

}
