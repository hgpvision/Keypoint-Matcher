/*
* 文件名称：Matcher.h
* 依赖库：opencv2411或其他版本
* 测试环境：VS2015 x86
* 摘 要：采用SAD度量尺度进行特征点匹配。
* 匹配策略：将整个图片分成区域（舍去周边部分像素，至少舍去sadBlockSize个像素，防止溢出），将所有候选特征点放入到不同区内，
*			以特征点所在区域为中心区域，先进行0级匹配，即以基准点所在区域在另一帧中对应区域内进行匹配，选择该区域中的最小SAD值，若小于
*			零级SAD阈值，则认为匹配成功，存入matches，不再进行1级和2级匹配；1级和2级类似，在另一帧中，以特征点所在区域偏移1和2个区域（总共8
*			和16个区域内进行匹配），1级匹配如果满足1级SAD阈值，则不再进行2级匹配；若2级结束后，依然未能满足指定2级SAD阈值，则放弃该点，视该点匹配失败，
*			最多有两级匹配。
* 说明：本匹配类开始是针对FAST特征写的，但实际上该Matcher与FAST没有任何关系，因此，只要是提取出KeyPoint类的特征，都可以使用该Matcher进行匹配
* 使用实例：见示例程序testMatcher.cpp
* 2016.05.23
*/

#pragma once
#include <opencv2/opencv.hpp>

class Matcher
{
public:
	//@Brief： FastMatcher匹配类的构造函数
	//@Input：	imgSize：处理图片的尺寸，用于确定分区
	//			regionSize：每个区域的大小，正方形边长
	//			sadBlockSize：计算SAD值的块的大小，只能取2或3，此为一半值，实际大小为2*sadBlockSize+1，默认为2
	//@Output： 无直接输出，直接改变类的属性变量，都是一些基本变量，起到初始化类的参数的作用
	Matcher(
		cv::Size imgSize, 
		int regionSize,
		int sadBlockSize =2);


	//@Brief：	匹配类的主函数
	//@Input：	prevImg：上一帧图像
	//			currImg：下一帧图像
	//			prevCorner：上一帧图像检测到的所有特征点
	//			currCorner：下一帧图像检测到的所有特征点
	//			matches：存储匹配结果
	//@Output：	无直接输出，直接改变引用参数matches的值，同时改变属性变量_prevORcurr的值
	void match(
		cv::Mat prevImg,
		cv::Mat currImg,
		std::vector<cv::KeyPoint> prevCorner,
		std::vector<cv::KeyPoint> currCorner,
		std::vector<cv::DMatch>& matches);


	//@Brief： 分区函数，将所有检测到的点，按照区域大小对整个图像进行分区，将各检测到的特征点放入到指定的区域内
	//@Input：	corner：要进行分区图片上所检测到的所有特征点
	//			cornerNum：即corner的尺寸，特征点的个数
	//@Output： 无直接输出，最终的结果直接改变属性变量_region
	void partition(std::vector<cv::KeyPoint> corner, size_t cornerNum);


	//@Brief：	以当前帧中的特征点作为基准点，在上一帧中匹配对应点。计算SAD值，按照指定规则选择特征点在另一帧中的匹配点（若匹配上）
	//@Input：	prev：上一帧图像
	//			curr：当前帧图像
	//			corner：要进行匹配的基准点（当前帧中的单个特征点）
	//			nextCorner：此参数不是指下一帧，而是指另一帧，这里是指上一帧的所有特征点，将在这些特征点中找出corner的匹配点
	//			matches：匹配结果，queryIdx恒指上一帧图像中的点编号（在prevCorner中的索引），trainIdx恒指下一帧图像中的点编号（在currCorner中的索引）
	//@Output：	无直接输出，直接改变matches引用参数的值
	void calcSADCurrBased(
		cv::Mat prev,
		cv::Mat curr,
		cv::KeyPoint corner, 
		size_t cornerID, 
		std::vector<cv::KeyPoint> nextCorner, 
		std::vector<cv::DMatch>& matches);


	//@Brief：	以上一帧中的特征点作为基准点，在当前帧中匹配对应点。计算SAD值，按照指定规则选择特征点在另一帧中的匹配点（若匹配上）
	//@Input：	prev：上一帧图像
	//			curr：当前帧图像
	//			corner：要进行匹配的基准点（上一帧中的单个特征点）
	//			nextCorner：此参数不是指下一帧，而是指另一帧，这里是指当前帧的所有特征点，将在这些特征点中找出corner的匹配点
	//			matches：匹配结果，queryIdx恒指上一帧图像中的点编号（在prevCorner中的索引），trainIdx恒指下一帧图像中的点编号（在currCorner中的索引）
	//@Output：	无直接输出，直接改变matches引用参数的值
	void calcSADPrevBased(
		cv::Mat prev, 
		cv::Mat curr,
		cv::KeyPoint corner,
		size_t cornerID,
		std::vector<cv::KeyPoint> nextCorner,
		std::vector<cv::DMatch>& matches);
private:
	//SAD阈值，分三个等级：0级别匹配即在基准点所在区域内匹配；1级匹配即在所在区域偏移1个区域（总共8个区域）内进行匹配；2级偏移两个区域（总共16个区域）
	double _sadThreshold0;		//0级匹配SAD阈值，最小
	double _sadThreshold1;		//1级匹配SAD阈值，中间
	double _sadThreshold2;		//2级匹配SAD阈值，最大
	double _sadInitVal;			//minSAD的初始值，在构造函数中将初始化为1000

	//计算SAD值的块的大小
	int _sadBlockSize;			//实际大小为2*_sadBlockSize+1

	int _regionSize;			//每个小区域的大小
	int _totalRegionNum;		//整个图片舍去部分像素后，所分成区域的总块数，等于_regionXY[0]*_regionXY[1]
	size_t _originXY[2];		//所有区域的原点坐标（即舍去部分像素后，真正考虑部分图像的原点坐标）
	size_t _endXY[2];			//所有区域的终点坐标（即舍去部分像素后，真正考虑部分图像的终点坐标，矩形的另一个点，与原点成对角关系）
	size_t _regionXY[2];		//在x,y方向所分成区域的个数

	//_region用来存储分区结果，该变量是一个二维vector容器，共有_totalRegionNum个矢量，对应每个区域（区域先按行排列）中的特征点的索引值
	std::vector<std::vector<size_t> > _region;		//将点分成区域后得到的矢量数组，该容器中的每个矢量存入的是被分区特征点的索引值，每个矢量元素个数不定

	//下面这个标志位是为了减小分区的次数，比如前一次匹配如果是对下帧进行分区，且下次匹配是对上帧图像进行分区，则，可以不用再分区了，直接利用上次匹配分区的结果
	bool _prevORcurr;								//标志位，上一次匹配是对上一帧分区（1）还是对下一帧分区（0）
};