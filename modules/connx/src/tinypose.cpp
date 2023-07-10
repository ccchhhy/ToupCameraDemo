#include "tinypose.h"

TinyPose::TinyPose()
{
    // CPU
    auto model_file =  "./models/TinyPose/model.pdmodel";
    auto params_file = "./models/TinyPose/model.pdiparams";
    auto config_file = "./models/TinyPose/infer_cfg.yml";

    auto option = fastdeploy::RuntimeOption();
    option.UseCpu();
    model = new fastdeploy::vision::keypointdetection::PPTinyPose(
        model_file, params_file, config_file, option);

    if (!model->Initialized()) {
        std::cerr << "Failed to initialize." << std::endl;
        return;
    }
}
TinyPose::TinyPose(bool isGPU)
{
    // GPU
    auto model_file =  "./models/TinyPose/model.pdmodel";
    auto params_file = "./models/TinyPose/model.pdiparams";
    auto config_file = "./models/TinyPose/infer_cfg.yml";

    auto option = fastdeploy::RuntimeOption();
    option.UseGpu();
    model = new fastdeploy::vision::keypointdetection::PPTinyPose(
        model_file, params_file, config_file, option);

    if (!model->Initialized()) {
        std::cerr << "Failed to initialize." << std::endl;
        return;
    }
}

void TinyPose::run(const cv::Mat input_image, cv::Mat &output_image)
{
    // 推理
    auto im = input_image;
    fastdeploy::vision::KeyPointDetectionResult res;
      if (!model->Predict(&im, &res)) {
        std::cerr << "TinyPose Prediction Failed." << std::endl;
        return;
      } else {
        std::cout << "TinyPose Prediction Done!" << std::endl;
      }

    std::cout << res.Str() << std::endl;
    auto vis_im = fastdeploy::vision::VisKeypointDetection(im, res, 0.5);
    output_image = vis_im;
    std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;

}


void TinyPose:: preProcessing(const cv::Mat& input_image, Ort::Value& input_tensor){

}


void TinyPose::postProcessing(Ort::Value& output_tensor, cv::Mat& output_image){

}
