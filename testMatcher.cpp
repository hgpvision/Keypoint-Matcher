/*
* 文件名称：testMatcher.cpp
* 依赖库：opencv2411或其他版本
* 测试环境：VS2015 x86
* 摘 要：测试Matcher匹配类
* 包含文件：ReadImages.h/.cpp，读入图片类；Matcher.h/.cpp匹配类，二者完全独立，后者是该测试程序的测试内容（测试匹配效果）
* 说明：本测试程序使用的是FAST进行特征提取，实际上Matcher匹配类不限所使用特征的种类，因为Matcher与FAST没有任何关系，
*		因此，只要是提取出KeyPoint类的特征，都可以使用该Matcher进行匹配
* 测试数据：测试数据包括三套，一套是KITTI 01数据集，共10帧，运动幅度比较大，_regionSize取30；另一套是ETH的数据集，共49帧，运动幅度较小，_regionSize取10；
			第三套是KITTI 02数据集，共101帧，运动幅度也比较大（但图片有些部分较小），窗口取20。
* 结果：选择合适的_regionSize对结果影响还是挺大的：对于运动幅度较小情况，窗口不宜太大（取一个较小值），不然会造成很多的误匹配，且浪费时间；对于运动幅度较大的情况，
*		窗口适当取大，但不要过大，过大不会继续增多成功匹配的点，反而可能增加误匹配点及运算时间
* 2016.05.23
*/

#include <iostream>
#include<opencv2/opencv.hpp>
#include "Matcher.h"
#include "ReadImages.h"

int main()
{
	//定义一个读入图片类
	//ReadImages imgReader("G:\\KITTI_Database\\SLAM\\01", "000", ".png");
	//ReadImages imgReader("G:\\ETH ASL\\ETH-IMAGE-MODIFIED\\Img1", "", ".png");
	ReadImages imgReader("G:\\KITTI_Database\\SLAM\\02", "0000", ".png");


	//定义一个FAST特征检测子
	cv::FastFeatureDetector fast(40);
	
	//声明存储fast角点的容器
	std::vector<cv::KeyPoint> corner1;
	std::vector<cv::KeyPoint> corner2;

	//读入第一帧图像（上一帧）
	//cv::Mat prev = imgReader.loadImage(101, 0);
	//cv::Mat prev = imgReader.loadImage(1, 0);
	cv::Mat prev = imgReader.loadImage(1, 0);

	//对第一帧图像检测fast角点
	fast.detect(prev, corner1);

	//当前帧
	cv::Mat curr;

	//两帧的备份，用来在上面画点，显示结果，同时不污染原图片
	cv::Mat prevCopy, currCopy;

	//声明存储匹配结果的变量
	std::vector<cv::DMatch> matches;

	//定义一个fast匹配类，并初始化，每块区域的大小为30，计算SAD的块大小为2*2+1，（第三个参数仅能为2或3）
	//Matcher matcher(prev.size(), 30, 2);
	//Matcher matcher(prev.size(), 10, 2);
	Matcher matcher(prev.size(), 20, 2);

	//连续读入图片，连续进行帧间匹配
	//for (int i = 102;i < 111;i++)
	//for (int i = 2;i < 50;i++)
	for (int i = 2;i < 100;i++)
	{
		//读入下一帧
		curr = imgReader.loadImage(i, 0);

		//复制图像，以备画点
		prev.copyTo(prevCopy);
		curr.copyTo(currCopy);

		//namedWindow("prev", 1);
		//namedWindow("curr", 1);
		//imshow("prev", prev);
		//imshow("curr", curr);
		//waitKey(2000);

		//fast特征检测
		fast.detect(curr, corner2);
		
		//每次匹配，记得清除上一次匹配
		matches.clear();
		matcher.match(prev, curr, corner1, corner2, matches);

		//匹配结果显示
		size_t matchesNum = matches.size();
		std::cout << "正在进行第" << i - 1 << "帧和第" << i << "帧匹配：" << std::endl;
		std::cout << "matchesNum= " << matchesNum << std::endl;
		for (int j = 0;j < matchesNum;j++)
		{
			cv::circle(prevCopy, corner1[matches[j].queryIdx].pt, 3, cv::Scalar(0, 0, 255), -1, 8);
			cv::circle(currCopy, corner2[matches[j].trainIdx].pt, 3, cv::Scalar(0, 0, 255), -1, 8);

			cv::namedWindow("prevCopy", 1);
			cv::namedWindow("currCopy", 1);
			cv::imshow("prevCopy", prevCopy);
			cv::imshow("currCopy", currCopy);
			cv::waitKey(10);
		}

		//当前帧检测到的角点，赋值给上一帧
		//corner1.clear();		//加了这句还是会出错
		//corner1 = corner2;
		//使用上面的直接赋值，有时会出现错误（在使用ETH数据集，进行到第17帧时，出错），原因不明：Expression:"(_Ptr_user & (_BIG_ALLOCATION_ALIGNMENT -1))==0" && 0
		//后来改用下面的两条（第二条可以不用）
		corner1.swap(corner2);
		corner2.clear();

		//当前帧赋值给上一帧
		curr.copyTo(prev);
	}

	std::cout << "所有帧完成！" << std::endl;
	system("pause");
}