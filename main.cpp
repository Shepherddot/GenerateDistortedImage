#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

int main() {
  cv::Mat K = (cv::Mat_<float>(3, 3) <<
      9.842439e+02, 0.000000e+00, 6.900000e+02,
      0.000000e+00, 9.808141e+02, 2.331966e+02,
      0.000000e+00, 0.000000e+00, 1.000000e+00);

  cv::Mat KB_K = (cv::Mat_<float>(3, 3) <<
      326.3346067368311, 0, 635.1286615099734,
      0, 326.7375607973489, 354.0297991203038,
      0, 0, 1);

  cv::Mat KB_D = (cv::Mat_<float>(4, 1) << 0.0546962, 0.0622869, -0.0500086, 0.0123447);

  cv::Mat original_image = cv::imread(
      "../data/0000000016.png",
      0);

  uchar* pPixel_o = original_image.data;

  cv::Size new_image_size;
  new_image_size.width = original_image.cols;
  new_image_size.height = original_image.rows * 2.5;
  cv::Mat distorted_image = cv::Mat::zeros(new_image_size, original_image.type());

  uchar* pPixel_d = distorted_image.data;

  for(int y = 0; y < original_image.rows; y++){
    for(int x = 0; x < original_image.cols; x++){
      cv::Mat original_pt = K.inv() * (cv::Mat_<float>(3, 1) << x, y, 1.0);

      cv::Mat normalized_pt = cv::Mat(1, 1, CV_32FC2);
      normalized_pt.at<cv::Vec2f>(0, 0)[0] = 10 * original_pt.at<float>(0) / original_pt.at<float>(2);
      normalized_pt.at<cv::Vec2f>(0, 0)[1] = 10 * original_pt.at<float>(1) / original_pt.at<float>(2);

      cv::Mat distorted_pt = cv::Mat(1, 1, CV_32FC2);

      cv::fisheye::distortPoints(normalized_pt, distorted_pt, KB_K, KB_D);

      int x_axis = distorted_pt.at<cv::Vec2f>(0, 0)[0];
      int y_axis = distorted_pt.at<cv::Vec2f>(0, 0)[1] + 100;

      if(x_axis >= 0 && x_axis < distorted_image.cols){
        if(y_axis >= 0 && y_axis < distorted_image.rows){
          *(pPixel_d + y_axis * distorted_image.cols + x_axis) = *(pPixel_o + y * original_image.cols + x);
        }
      }

    }
  }

  cv::imshow("original image", original_image);
  cv::imshow("distorted_image", distorted_image);
  cv::waitKey();

  return 0;
}
