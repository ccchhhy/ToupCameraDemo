#ifndef TINYPOSE_H
#define TINYPOSE_H

#include <connx.h>
#include <fastdeploy/vision.h>

class TinyPose : public COnnx
{
public:
    TinyPose();
    TinyPose(bool isGPU=false);
    ~TinyPose();
    void run(const cv::Mat input_image, cv::Mat& output_image) override;
    void preProcessing(const cv::Mat& input_image, Ort::Value& input_tensor) override;
    void postProcessing(Ort::Value& output_tensor, cv::Mat& output_image) override;
private:
    fastdeploy::vision::keypointdetection::PPTinyPose* model;
};

#endif // TINYPOSE_H
