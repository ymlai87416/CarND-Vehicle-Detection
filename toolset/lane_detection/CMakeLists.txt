cmake_minimum_required(VERSION 3.9)
project(lane_detection)

find_package( OpenCV REQUIRED )
find_package( Qt5Core REQUIRED)
find_package( Qt5Gui REQUIRED)
find_package( Qt5Widgets REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 11)

add_executable( birdView birdView.cpp CameraCalibrator.cpp)
add_executable( featureExtraction featureExtraction.cpp)
add_executable( laneDetection laneDetection.cpp)

# link libraries
target_link_libraries( birdView ${OpenCV_LIBS})
target_link_libraries( featureExtraction ${OpenCV_LIBS})
target_link_libraries( laneDetection ${OpenCV_LIBS})
target_link_libraries(birdView Qt5::Core)
target_link_libraries(birdView Qt5::Gui)
target_link_libraries(birdView Qt5::Widgets)

# copy required images to every directory with executable
FILE(COPY ${CMAKE_SOURCE_DIR}/videos/ DESTINATION ./)
FILE(COPY ${CMAKE_SOURCE_DIR}/videos/ DESTINATION "Debug/")
FILE(COPY ${CMAKE_SOURCE_DIR}/videos/ DESTINATION "Release/")
FILE(COPY ${CMAKE_SOURCE_DIR}/camera_cal/ DESTINATION ./camera_cal/)
FILE(COPY ${CMAKE_SOURCE_DIR}/camera_cal/ DESTINATION "Debug/camera_cal/")
FILE(COPY ${CMAKE_SOURCE_DIR}/camera_cal/ DESTINATION "Release/camera_cal/")