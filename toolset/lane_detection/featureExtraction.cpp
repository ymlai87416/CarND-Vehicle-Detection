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
bool mustInitBirdviewTransform = true;
cv::Mat lambda( 2, 4, CV_32FC1 );

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
    cameraCalibrator.setCalibrationFlag(false,false);
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
    // Get the Perspective Transform Matrix i.e. lambda
    if(mustInitBirdviewTransform){
        std::cout<< "w " << img.cols << " h " << img.rows << std::endl;
        // Input Quadilateral or Image plane coordinates
        cv::Point2f inputQuad[4];
        // Output Quadilateral or World plane coordinates
        cv::Point2f outputQuad[4];

        // Set the lambda matrix the same type and size as input
        lambda = cv::Mat::zeros( img.rows, img.cols, img.type() );

        // The 4 points that select quadilateral on the input , from top-left in clockwise order
        // These four pts are the sides of the rect box used as input
        inputQuad[0] = cv::Point2f( 590, 450 );
        inputQuad[1] = cv::Point2f( 689, 450 );
        inputQuad[2] = cv::Point2f( 1135, 720 );
        inputQuad[3] = cv::Point2f( 189, 720 );
        // The 4 points where the mapping is to be done , from top-left in clockwise order
        outputQuad[0] = cv::Point2f( 315, 0 );
        outputQuad[1] = cv::Point2f( 960, 0 );
        outputQuad[2] = cv::Point2f( 960, 720 );
        outputQuad[3] = cv::Point2f( 315, 720 );
        lambda = cv::getPerspectiveTransform( inputQuad, outputQuad );
        mustInitBirdviewTransform = false;
    }

    // Apply the Perspective Transform just found to the src image
    warpPerspective(img, out, lambda, img.size() );
}

void absSobelThreshold(cv::UMat& l_channel, cv::UMat& out, char orient, int threshMin, int threshMax, int blurFilterSize){
    cv::UMat gray, absSobel, scaledAbsSobel;
    double sobelMin, sobelMax;

    cv::GaussianBlur(l_channel, gray, cv::Size(blurFilterSize, blurFilterSize), 0);

    if (orient == 'x')
        cv::Sobel(gray, absSobel, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    else
        cv::Sobel(gray, absSobel, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

    cv::minMaxLoc(absSobel, &sobelMin, &sobelMax);

    cv::convertScaleAbs( absSobel, scaledAbsSobel, 255.0 / std::max(abs(sobelMax), abs(sobelMin)), 0);

    cv::threshold(scaledAbsSobel, scaledAbsSobel, threshMin, 255, cv::THRESH_TOZERO );
    cv::threshold(scaledAbsSobel, scaledAbsSobel, threshMax, 255 ,cv::THRESH_TOZERO_INV );
    cv::threshold(scaledAbsSobel, out, 0, 255, cv::THRESH_BINARY);

    out.convertTo(out, l_channel.type());
}

void yellowColor(cv::UMat b_channel, cv::UMat& out, int lowerYellow, int upperYellow){
    cv::threshold( b_channel, b_channel, lowerYellow, 255, cv::THRESH_TOZERO );
    cv::threshold( b_channel, b_channel, upperYellow, 255 ,cv::THRESH_TOZERO_INV );
    cv::threshold( b_channel, out, 0, 255, cv::THRESH_BINARY);
}

void whiteColor(cv::UMat l_channel, cv::UMat& out, int lowerWhite){
    cv::threshold( l_channel, l_channel, lowerWhite, 255, cv::THRESH_TOZERO );
    cv::threshold( l_channel, l_channel, 255, 255 ,cv::THRESH_TOZERO_INV );
    cv::threshold( l_channel, out, 0, 255, cv::THRESH_BINARY);
}

void extractFeature(cv::UMat& img, cv::UMat& out){
    int lowerSobel = 20;
    int upperSobel = 100;
    int lowerWhite = 215;
    int lowerYellow = 155;
    int upperYellow = 200;

    cv::UMat uImg_lab;
    cv::cvtColor(img, uImg_lab, cv::COLOR_BGR2Lab);
    cv::Mat img_lab = uImg_lab.getMat(cv::ACCESS_READ);
    cv::Mat channels[3];
    cv::UMat b_channel, l_channel;
    cv::split(img_lab, channels);
    channels[0].copyTo(l_channel);
    channels[2].copyTo(b_channel);
    img_lab.release();

    cv::UMat sobel_out, white_out, yellow_out, result;
    result = cv::UMat::zeros(channels[0].rows, channels[0].cols, channels[0].type());

    absSobelThreshold(l_channel, sobel_out, 'x', lowerSobel, upperSobel, 13);
    yellowColor(b_channel, yellow_out, lowerYellow, upperYellow);
    whiteColor(l_channel, white_out, lowerWhite);

    cv::Mat finalImage;
    sobel_out.copyTo(channels[0]);
    yellow_out.copyTo(channels[1]);
    white_out.copyTo(channels[2]);
    cv::merge(channels, 3, finalImage);

    finalImage.copyTo(out);
}

void pipeline(cv::Mat& img, cv::Mat& out){
    cv::Size newSize(static_cast<int>(img.cols), static_cast<int>(img.rows));
    cv::UMat uImg, uOut, uOut2;
    img.copyTo(uImg);
    //uImg = cameraCalibrator.remap(uImg, newSize);
    //canny(uImg, uOut);
    transformBirdView(uImg, uOut2);
    extractFeature(uOut2, uOut);
    //uOut = uImg;
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