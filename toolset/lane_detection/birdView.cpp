#include <string>
#include <iostream>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <QtWidgets/QFileDialog>
#include <QtCore/QString>
#include <QtWidgets/QApplication>

#include "videoprocessor.h"
#include "CameraCalibrator.h"
#include <fstream>

CameraCalibrator cameraCalibrator;

void calibrateCamera(){
    cv::Mat image;
    std::vector<std::string> filelist;

    for (int i=1; i<=20; i++) {
        std::stringstream str;
        str << "camera_cal/calibration" << i << ".jpg";
        std::cout << str.str() << std::endl;
        image= cv::imread(str.str(),0);

        filelist.push_back(str.str());
    }

    // Create calibrator object

    // add the corners from the chessboard
    cv::Size boardSize(9,6);
    cameraCalibrator.addChessboardPoints(
            filelist,	// filenames of chessboard image
            boardSize, "Detected points");	// size of chessboard

    // calibrate the camera
    cameraCalibrator.setCalibrationFlag(true,true);
    cameraCalibrator.calibrate(image.size());

    // Exampple of Image Undistortion
    image = cv::imread(filelist[14],0);
    cv::Size newSize(static_cast<int>(image.cols*1.5), static_cast<int>(image.rows*1.5));
    cv::Mat uImage= cameraCalibrator.remap(image, newSize);

    // display camera matrix
    cv::Mat cameraMatrix= cameraCalibrator.getCameraMatrix();
    std::cout << " Camera intrinsic: " << cameraMatrix.rows << "x" << cameraMatrix.cols << std::endl;
    std::cout << cameraMatrix.at<double>(0,0) << " " << cameraMatrix.at<double>(0,1) << " " << cameraMatrix.at<double>(0,2) << std::endl;
    std::cout << cameraMatrix.at<double>(1,0) << " " << cameraMatrix.at<double>(1,1) << " " << cameraMatrix.at<double>(1,2) << std::endl;
    std::cout << cameraMatrix.at<double>(2,0) << " " << cameraMatrix.at<double>(2,1) << " " << cameraMatrix.at<double>(2,2) << std::endl;

    // Store everything in a xml file
    cv::FileStorage fs("calib.xml", cv::FileStorage::WRITE);
    fs << "Intrinsic" << cameraMatrix;
    fs << "Distortion" << cameraCalibrator.getDistCoeffs();
}

// processing function
void canny(cv::UMat& img, cv::UMat& out) {

    // Convert to gray
    if (img.channels()==3)
        cv::cvtColor(img,out,cv::COLOR_BGR2GRAY);

    // Compute Canny edges
    cv::Canny(out,out,100,200);
    // Invert the image
    cv::threshold(out,out,128,255,cv::THRESH_BINARY_INV);

}

void transformBirdView(cv::UMat& img, cv::UMat& out){
// Input Quadilateral or Image plane coordinates
    Point2f inputQuad[4];
    // Output Quadilateral or World plane coordinates
    Point2f outputQuad[4];

    // Lambda Matrix
    Mat lambda( 2, 4, CV_32FC1 );
    //Input and Output Image;
    Mat input, output;

    //Load the image
    input = imread( "lena.jpg", 1 );
    // Set the lambda matrix the same type and size as input
    lambda = Mat::zeros( input.rows, input.cols, input.type() );

    // The 4 points that select quadilateral on the input , from top-left in clockwise order
    // These four pts are the sides of the rect box used as input
    inputQuad[0] = Point2f( -30,-60 );
    inputQuad[1] = Point2f( input.cols+50,-50);
    inputQuad[2] = Point2f( input.cols+100,input.rows+50);
    inputQuad[3] = Point2f( -50,input.rows+50  );
    // The 4 points where the mapping is to be done , from top-left in clockwise order
    outputQuad[0] = Point2f( 0,0 );
    outputQuad[1] = Point2f( input.cols-1,0);
    outputQuad[2] = Point2f( input.cols-1,input.rows-1);
    outputQuad[3] = Point2f( 0,input.rows-1  );

    // Get the Perspective Transform Matrix i.e. lambda
    lambda = getPerspectiveTransform( inputQuad, outputQuad );
    // Apply the Perspective Transform just found to the src image
    warpPerspective(input,output,lambda,output.size() );
}

void pipeline(cv::Mat& img, cv::Mat& out){
    cv::Size newSize(static_cast<int>(img.cols), static_cast<int>(img.rows));
    cv::Mat uImage= cameraCalibrator.remap(img, newSize);

    cv::UMat uImg, uOut;
    uImage.copyTo(uImg);
    canny(uImg, uOut);
    uOut.copyTo(out);
}

class BirdViewApp : public QApplication
{
public:

    BirdViewApp(int argc, char** argv) : QApplication(argc, argv){}


	QString openFile()
	{
		QString filename = QFileDialog::getOpenFileName(
			0,
			"Open Document",
			QDir::currentPath(),
			"Video Files (*.mp4 *.avi)");

		return filename;
	}

    void run() {
        std::ifstream f("calib.xml");
        if (f.good()) {
            cv::Mat cameraMatrix;
            cv::Mat cameraDistCoeffs;
            cv::FileStorage fs("calib.xml", cv::FileStorage::READ);
            fs["Intrinsic"] >> cameraMatrix;
            fs["Distortion"] >> cameraDistCoeffs;
            cameraCalibrator.setCameraMatrix(cameraMatrix);
            cameraCalibrator.setDistCoeffs(cameraDistCoeffs);
        }
        else
            calibrateCamera();

        std::string fileName = openFile().toStdString();

        // Open the video file
        cv::VideoCapture capture(fileName);

        if (!capture.isOpened())
            return ;

        // Get the frame rate
        double rate= capture.get(cv::CAP_PROP_FPS);
        std::cout << "Frame rate: " << rate << "fps" << std::endl;

        bool stop(false);
        cv::Mat frame; // current video frame
        cv::namedWindow("Extracted Frame");

        // Now using the VideoProcessor class

        // Create instance
        VideoProcessor processor;

        // Open video file
        processor.setInput(fileName);

        // Declare a window to display the video
        processor.displayInput("Input Video");
        processor.displayOutput("Output Video");

        // Play the video at the original frame rate
        processor.setDelay(1000./processor.getFrameRate());

        // Set the frame processor callback function
        processor.setFrameProcessor(pipeline);

        // output a video
        processor.setOutput("output.mp4",-1,0 );

        // Start the process
        processor.run();
    }

};

int main(int argc, char** argv) {
	BirdViewApp app(argc, argv);
    app.run();
}