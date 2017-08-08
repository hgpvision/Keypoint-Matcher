/*
* 文件名称：ReadImages.h
* 摘 要：读取图片类，图片序列放入到一个文件夹中，如"C:\\imgFile"，按一定格式命名，比如"0001.jpg","00010.jpg","1.jpg"
* 使用实例：
* 包含头文件：			#include "ReadImages.h"
* 先声明要给读入图片类：ReadImages imageReader("C:\\imgFile", "", ".jpg");
* 读入图片：			Mat prev = imageReader.loadImage(1,0); //将读入"1.jpg"图片，灰度格式
* 2016.05.23
*/

#pragma once
#include <opencv2/opencv.hpp>

typedef struct ImgSource
{
	std::string _basepath = "";		//图片基本路径
	std::string _imagename = "";	//图片名（不含编号）
	std::string _suffix = "";		//图片后缀
}ImageSource;

class ReadImages
{
public:
	ReadImages() {}

	//@Brief：	读取图片类构造函数
	//@Input：	basepath：图片所在文件夹的基本路径（不含最后的\\）
	//			imagename：图片的名字（不含编号），例如图片放在文件夹"C:\\imgFile"下，命名为0001.jpg,0002.jpg,...,00010.jpg，则imagename为"000",1,2,10为编号
	//			suffix：图片后缀，如上，则为".jpg"
	ReadImages(const std::string basepath, const std::string imagename, const std::string suffix);

	//@Brief：	读入图片，调用imread()函数
	//@Input：	imgId：图像编号
	//			imgType：读入图片格式，和imread()统一，0表示读入灰度图，1表示读入彩色图
	//@Output：	输出读入的图片，Mat格式
	cv::Mat loadImage(int imgId, int imgType);	//读入图片
private:
	ImageSource _imgSource;			//图片路径信息
	cv::Rect _roi;
};
