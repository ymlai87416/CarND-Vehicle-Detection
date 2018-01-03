//
// Created by ymlai on 2/1/2018.
//

#ifndef LANE_DETECTION_LINE_H
#define LANE_DETECTION_LINE_H

#endif //LANE_DETECTION_LINE_H

class Line{
private:
    bool detected = false;
    double bestx;
    double* bestFit;
    double* currentFit;
    double radiusOfCurvature;
    double lineBasePos;
    double diffs[3];
    double* allx;
    double* ally;
};
