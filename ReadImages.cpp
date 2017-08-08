/*
* 文件名称：ReadImages.cpp
* 摘 要：读取图片类，图片序列放入到一个文件夹中，如"C:\\imgFile"，按一定格式命名，比如"0001.jpg","00010.jpg","1.jpg"
* 使用实例：
* 包含头文件：			#include "ReadImages.h"
* 先声明要给读入图片类：ReadImages imageReader("C:\\imgFile", "", ".jpg");
* 读入图片：			Mat prev = imageReader.loadImage(1,0); //将读入"1.jpg"图片，灰度格式
* 2016.05.23
*/

#include "ReadImages.h"
#pragma once

ReadImages::ReadImages(std::string basepath, const std::string imagename, const std::string suffix)
{
	//将路径中的文件夹分隔符统一为'/'（可以不需要这个for循环，会自动统一，该语句使用的命令参考C++ Primer）
	for (auto &c : basepath)
	{
		if (c == '\\')
		{
			c = '/';
		}
	}

	_imgSource._basepath = basepath + "/";	//这里采用'/'，而不是'\\'
	_imgSource._imagename = imagename;		//图片名（不含编号）
	_imgSource._suffix = suffix;				//图像扩展名

}

//读入单张图片
//输入：imgId，一般要读入序列图片，图片都有个编号
cv::Mat ReadImages::loadImage(int imgId, int imgType)
{	
	//将图片编号转好字符串
	std::stringstream ss;
	std::string imgNum;
	ss << imgId;
	ss >> imgNum;

	//得到图片的完整绝对路径
	std::string path = _imgSource._basepath + _imgSource._imagename + imgNum + _imgSource._suffix;
	
	cv::Mat img = cv::imread(path,imgType);

	return img;
}