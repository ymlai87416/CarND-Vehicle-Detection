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
#include "line.h"
#include <tuple>

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

std::tuple<int, int> findLeftAndRightBase(cv::Mat& binaryWarp){
    cv::Mat imageBottom = binaryWarp(cv::Rect(0, binaryWarp.rows*3/4, binaryWarp.cols, binaryWarp.rows/4));
    cv::Mat histogram = cv::Mat(1, binaryWarp.cols, CV_64FC1);
    cv::reduce(binaryWarp, histogram, 0, CV_REDUCE_SUM);

    cv::Mat leftHistogram = histogram(cv::Rect(0, 0, histogram.cols/2, 1));
    cv::Mat rightHistogram = histogram(cv::Rect(histogram.cols/2, 0, histogram.cols - (histogram.cols/2), 1));
    double max, min;
    cv::Point minLoc, maxLoc;
    int leftX, rightX;
    cv::minMaxLoc(leftHistogram, &min, &max, &minLoc, &maxLoc);
    leftX = maxLoc.x;
    cv::minMaxLoc(rightHistogram, &min, &max, &minLoc, &maxLoc);
    rightX = maxLoc.x;

    std::tuple<cv::Point, cv::Point> result = std::make_tuple(leftX, rightX);
}


void findLeftAndRightLane(cv::Mat& binaryWarp, Line left_lane, Line right_lane){
    int noDetectWindow = 27;
    int windowWidth=50;
    int windowHeight = binaryWarp.rows / noDetectWindow;
    int windowMargin = 100;
    int minPix = 400;
    int expectedLaneWidth=3.7;
    int owerLaneWidt=0.75;


    std::vector<cv::Point> locations;
    cv::findNonZero(binaryWarp, locations);

    std::tuple centers = findLeftAndRightBase(binaryWarp);

    left_lane_inds = []
    right_lane_inds = []
    left_windows = []
    right_windows = []
    is_left_window_good = []
    is_right_window_good = []
    leftw = []
    rightw = []
    consecutive_bad_left = 0
    consecutive_bad_right = 0
    current_leftw = 1
    current_rightw = 1

    win_y_low = int(binary_warped.shape[0]-window_height)
    win_y_high = int(binary_warped.shape[0])
    win_xleft_low = l_center - window_width//2
    win_xleft_high  = l_center + window_width//2
    win_xright_low = r_center - window_width//2
    win_xright_high = r_center + window_width//2
    left_windows.append((win_xleft_low, win_y_low, win_xleft_high, win_y_high))
    right_windows.append((win_xright_low, win_y_low, win_xright_high, win_y_high))


    for level in range(1,(int)(binary_warped.shape[0]/window_height)):
    win_y_low = int(binary_warped.shape[0]-(level+1)*window_height)
    win_y_high = int(binary_warped.shape[0]-level*window_height)

    image_layer = np.sum(binary_warped[win_y_low:win_y_high,:], axis=0)
    conv_signal = np.convolve(window, image_layer)

    offset = window_width//2
    l_min_index = int(max(l_center+offset-margin,0))
    l_max_index = int(min(l_center+offset+margin,binary_warped.shape[1]))
    l_order = self.create_reorder_array(l_min_index, l_max_index)
    l_conv_signal = (conv_signal[l_min_index:l_max_index])[l_order]
    l_center_t = np.argmax(l_conv_signal)
    l_center_t = l_order[l_center_t]+l_min_index-offset

    r_min_index = int(max(r_center+offset-margin,0))
    r_max_index = int(min(r_center+offset+margin,binary_warped.shape[1]))
    r_order = self.create_reorder_array(r_min_index, r_max_index)
    r_conv_signal = (conv_signal[r_min_index:r_max_index])[r_order]
    r_center_t = np.argmax(r_conv_signal)
    r_center_t = r_order[r_center_t]+r_min_index-offset


    win_xleft_low = l_center_t - offset
    win_xleft_high  = l_center_t + offset
    win_xright_low = r_center_t - offset
    win_xright_high = r_center_t + offset

    left_windows.append((win_xleft_low, win_y_low, win_xleft_high, win_y_high))
    right_windows.append((win_xright_low, win_y_low, win_xright_high, win_y_high))

    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    if len(good_left_inds) > minpix:
    is_left_window_good.append(True)
    l_center = l_center_t
    consecutive_bad_left = 0
    else:
    is_left_window_good.append(False)
    consecutive_bad_left = consecutive_bad_left + 1

    if len(good_right_inds) > minpix:
    is_right_window_good.append(True)
    r_center = r_center_t
    consecutive_bad_right = 0
    else:
    is_right_window_good.append(False)
    consecutive_bad_right = consecutive_bad_right + 1

    if(consecutive_bad_left == 1 and
       (l_center < 1.5 * window_width or l_center > binary_warped.shape[1] - 1.5 * window_width)):
    current_leftw = 0
    if(consecutive_bad_right == 1 and
       (r_center < 1.5 * window_width or r_center > binary_warped.shape[1] - 1.5 * window_width)):
    current_rightw = 0

    if(consecutive_bad_left == 2):
    current_leftw = current_leftw / 2
    elif(consecutive_bad_left == 4):
    current_leftw = 0

    if(consecutive_bad_right == 2):
    current_rightw = current_rightw / 2
    elif(consecutive_bad_right == 4):
    current_rightw = 0

    leftw_batch = binary_warped[nonzeroy[good_left_inds], nonzerox[good_left_inds]] * current_leftw
    rightw_batch = binary_warped[nonzeroy[good_right_inds], nonzerox[good_right_inds]] * current_rightw

    leftw.append(leftw_batch)
    rightw.append(rightw_batch)


# Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftw = np.concatenate(leftw)
    rightw = np.concatenate(rightw)

# Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


# reverse the bird view and the points here
    pts_left = np.transpose(np.vstack([leftx, lefty]))
    pts_right = np.transpose(np.vstack([rightx, righty]))

    pts_left = self.camera.transform_further_point_inv(pts_left, direction)
    pts_right = self.camera.transform_further_point_inv(pts_right, direction)

# Fit a second order polynomial to each
    left_fit = np.polyfit(pts_left[:, 1], pts_left[:, 0], 2, w=leftw)
    right_fit = np.polyfit(pts_right[:, 1], pts_right[:, 0], 2, w=rightw)

    if(self.__debug_flag):
# add the line fitting result here
    binary_warped[binary_warped>0] = 255
    binary_warped = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2RGB)

# draw the left rectangle
    for rect, is_good in zip(left_windows, is_left_window_good):
    if is_good:
        rect_color = (0, 255, 0)
    else:
    rect_color = (255, 0, 0)
    cv2.rectangle(binary_warped, (rect[0], rect[1]), (rect[2], rect[3]), rect_color, 2)

# draw the right rectangle
    for rect, is_good in zip(right_windows, is_right_window_good):
    if is_good:
        rect_color = (0, 255, 0)
    else:
    rect_color = (255, 0, 0)
    cv2.rectangle(binary_warped, (rect[0], rect[1]), (rect[2], rect[3]), rect_color, 2)

    if direction > 0:
    cv2.putText(binary_warped, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    elif direction < 0:
    cv2.putText(binary_warped, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    else:
    cv2.putText(binary_warped, "CENTER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    self.__debug_dict['line_trace'] = binary_warped


    return left_fit, right_fit, pts_left, pts_right, left_windows, \
                right_windows, is_left_window_good, is_right_window_good
}

void findLeftAndRightLanePrevLane(cv::Mat& img, Line left_lane, Line right_lane){}

void sanityCheck(Line left_lane, Line right_lane, cv::Size imgSize){}

void detectLane(cv::Mat& featureImg){}

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