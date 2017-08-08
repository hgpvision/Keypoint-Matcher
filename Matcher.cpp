/*
* 文件名称：Matcher.cpp
* 依赖库：opencv2411或其他版本
* 测试环境：VS2015 x86
* 摘 要：采用SAD度量尺度进行特征点匹配（SAD：计算块内所有像素对应灰度值差的绝对值的和）
* 匹配策略：将整个图片分成区域（舍去周边部分像素，至少舍去sadBlockSize个像素，防止溢出），将所有候选特征点放入到不同区内，
*			以特征点所在区域为中心区域，先进行0级匹配，即以基准点所在区域在另一帧中对应区域内进行匹配，选择该区域中的最小SAD值，若小于
*			零级SAD阈值，则认为匹配成功，存入matches，不再进行1级和2级匹配；1级和2级类似，在另一帧中，以特征点所在区域偏移1和2个区域（总共8
*			和16个区域内进行匹配），1级匹配如果满足1级SAD阈值，则不再进行2级匹配；若2级结束后，依然未能满足指定2级SAD阈值，则放弃该点，视该点匹配失败，
*			最多有两级匹配。
* 说明：本匹配类开始是针对FAST特征写的，但实际上该Matcher与FAST没有任何关系，因此，只要是提取出KeyPoint类的特征，都可以使用该Matcher进行匹配
* 使用实例：见示例程序testMatcher.cpp
* 2016.05.23
*/

#include "Matcher.h"

Matcher::Matcher(cv::Size imgSize, int regionSize, int sadBlockSize)
{
	_regionSize = regionSize;
	_sadBlockSize = sadBlockSize;

	size_t marginX;
	size_t marginY;
	size_t cols = imgSize.width;
	size_t rows = imgSize.height;

	//舍去一定像素再来考虑整除_regionSize的问题，必须保证至少四周都有_sadBlockSize的舍弃，防止在边缘区域的且在区域边缘的点在计算SAD值时，超出图像边界
	size_t cols_croped = imgSize.width - 2*_sadBlockSize;
	size_t rows_croped = imgSize.height - 2*_sadBlockSize;

	//获取剩余图像能够分成的区域数
	_regionXY[0] = cols_croped / _regionSize;
	_regionXY[1] = rows_croped / _regionSize;

	_totalRegionNum = _regionXY[0] * _regionXY[1];

	//因为要对剩余图像整分，所以剩余图像四周可能还有一些要舍弃的像素（margin，空白区）
	marginX = cols_croped % _regionSize;
	marginY = rows_croped % _regionSize;

	//根据图像左边和上边总共舍弃的像素量，来确定最终计算区域图像的原点坐标
	_originXY[0] = marginX / 2 - 1 + _sadBlockSize;
	_originXY[1] = marginY / 2 - 1 + _sadBlockSize;

	//根据图像右边和下边总共舍弃的像素量，来确定最终计算区域图像的终点坐标
	//_endXY[0] = cols - (marginX - marginX / 2 - 1) - _sadBlockSize;
	//_endXY[1] = rows - (marginY - marginX / 2 - 1) - _sadBlockSize;
	_endXY[0] = cols - (marginX - marginX / 2 )- 1 - _sadBlockSize;
	_endXY[1] = rows - (marginY - marginY / 2 )- 1 - _sadBlockSize;

	_prevORcurr = 1;

	_sadInitVal = 1000.0;

	//选用不同的块尺寸，计算SAD大小。块尺寸，所指定的SAD阈值当然不同（块内像素个数不同），本匹配类只考虑尺寸为5和7的SAD块（对应_sadBlockSize为2和3）
	switch (sadBlockSize)
	{
	case 2:
		//_sadThreshold0 = 100;
		//_sadThreshold1 = 200;
		//_sadThreshold2 = 300;
		_sadThreshold0 = 150;
		_sadThreshold1 = 200;
		_sadThreshold2 = 300;
		break;
	case 3:
		_sadThreshold0 = 300;
		_sadThreshold1 = 500;
		_sadThreshold2 = 700;
		break;
	default:
		assert(0);
	}
	//_region = std::vector<std::vector<size_t> >(_regionXY[0] * _regionXY[1], std::vector<size_t>(30));
}


void Matcher::match(
	cv::Mat prevImg, 
	cv::Mat currImg, 
	std::vector<cv::KeyPoint> prevCorner, 
	std::vector<cv::KeyPoint> currCorner, 
	std::vector<cv::DMatch>& matches)
{
	size_t prevCornerNum = prevCorner.size();
	size_t currCornerNum = currCorner.size();

	//如果上一帧检测到的点少，则对下一帧分区，以上一帧中的点作为基准点
	if (prevCornerNum <= currCornerNum)
	{
		//如果需要分区，需要对_region重新初始化
		_region = std::vector<std::vector<size_t> >(_totalRegionNum);
		partition(currCorner, currCornerNum);
		_prevORcurr = 0;		//因为本次是对下一帧分区，置_prevORcurr为0

		for (int i = 0;i < prevCornerNum;i++)
		{
			calcSADPrevBased(prevImg, currImg, prevCorner[i], i, currCorner, matches);
		}
	}
	else
	{
		//如果下一帧检测到的点少，则对上一帧分区，以下一帧中的点作为基准点
		//如果上一次匹配，是对下帧进行分区，那么这次匹配就不用再分区了，直接用之前的分区；如果不是，则需要重新进行分区
		if (_prevORcurr)
		{
			//如果需要分区，需要对_region重新初始化
			_region = std::vector<std::vector<size_t> >(_totalRegionNum);
			partition(prevCorner, prevCornerNum);
		}
		_prevORcurr = 1;			//因为是对上一帧分区，置为1
		for (size_t i = 0;i < currCornerNum;i++)
		{
			calcSADCurrBased(prevImg, currImg, currCorner[i], i, prevCorner,matches);
		}
	}
}


void Matcher::partition(std::vector<cv::KeyPoint> corner, size_t cornerNum)
{
	size_t xx, yy, index;

	for (size_t i = 0;i < cornerNum;i++)
	{
		//在margin区的点不在块内，舍去（也是为了防止溢出，因为计算SAD值，有可能会超出图像边缘，所以舍去一定的像素）
		if (corner[i].pt.x < _originXY[0] || corner[i].pt.y < _originXY[1] || 
			corner[i].pt.x >= _endXY[0] || corner[i].pt.y >= _endXY[1])
		{
			continue;
		}

		xx = (corner[i].pt.x - _originXY[0]) / _regionSize;
		yy = (corner[i].pt.y - _originXY[1]) / _regionSize;
		index = yy*_regionXY[0] + xx;

		_region[index].push_back(i);
	}
}


void Matcher::calcSADCurrBased(
	cv::Mat prev, cv::Mat curr, 
	cv::KeyPoint corner, 
	size_t cornerID, 
	std::vector<cv::KeyPoint> nextCorner, 
	std::vector<cv::DMatch>& matches)
{
	int xx = (corner.pt.x - _originXY[0]) / _regionSize;
	int yy = (corner.pt.y - _originXY[1]) / _regionSize;

	//不考虑哪些不再区域内的基准点
	if (xx < 0 || xx >= _regionXY[0] || yy < 0 || yy >= _regionXY[1]) { return; }
	
	cv::Mat prevPatch, currPatch;
	cv::DMatch matched;

	//本匹配程序中，虽然根据两帧图像检测到的点的数目不同，分为以上一帧为基准点和当前帧为基准点，但这只是为了实现上的简单，
	//习惯上，还是以上一帧检测出来的角点为基准点，然后在当前帧中找对应点，因此，恒以上一帧中的点为queryIdx，当前帧中的点为trainIdx
	matched.trainIdx = cornerID;

	curr(cv::Range(corner.pt.y - _sadBlockSize, corner.pt.y + _sadBlockSize + 1),
		cv::Range(corner.pt.x - _sadBlockSize, corner.pt.x + _sadBlockSize + 1)).copyTo(prevPatch);

	double minSAD = _sadInitVal;
	//cv::KeyPoint matchedCorner;

	//零级搜索
	int index = yy*_regionXY[0] + xx;
	int pointsNum = _region[index].size();
	for (int i = 0;i < pointsNum;i++)
	{
		prev(cv::Range(nextCorner[_region[index][i]].pt.y - _sadBlockSize, nextCorner[_region[index][i]].pt.y + _sadBlockSize + 1),
			cv::Range(nextCorner[_region[index][i]].pt.x - _sadBlockSize, nextCorner[_region[index][i]].pt.x + _sadBlockSize + 1)).copyTo(currPatch);
		double sad = cv::norm(prev, curr, cv::NORM_L1);
		if (sad < minSAD)
		{
			minSAD = sad;
			//matchedCorner = nextCorner[_region[index][i]];
			matched.queryIdx = _region[index][i];
		}
	}
	//穷尽一级搜索，取得SAD最小值后，再与指定阈值比较
	if (minSAD <= _sadThreshold0)
	{
		matches.push_back(matched);
		return;
	}

	//一级搜索：周边8个区域
	int index1[8];
	index1[0] = yy*_regionXY[0] + xx + 1;
	index1[1] = (yy + 1)*_regionXY[0] + xx + 1;
	index1[2] = (yy + 1)*_regionXY[0] + xx;
	index1[3] = (yy + 1)*_regionXY[0] + xx - 1;
	index1[4] = yy*_regionXY[0] + xx - 1;
	index1[5] = (yy - 1)*_regionXY[0] + xx - 1;
	index1[6] = (yy - 1)*_regionXY[0] + xx;
	index1[7] = (yy - 1)*_regionXY[0] + xx + 1;
	for (int i = 0;i < 8;i++)
	{
		//对于处于边缘上的基准点进行额外处理（边缘上的基准点搜索区域没有这么多）
		if (xx - 1 < 0 && (i > 2 && i < 6)) { continue; }
		if (xx + 1 == _regionXY[0] && (i == 0 || i == 1 || i == 7)) { continue; }
		if (yy - 1 < 0 && i > 4) { continue; }
		if (yy + 1 == _regionXY[1] && i < 4) { continue; }

		pointsNum = _region[index1[i]].size();
		for (int j = 0;j < pointsNum;j++)
		{
			prev(cv::Range(nextCorner[_region[index1[i]][j]].pt.y - _sadBlockSize, nextCorner[_region[index1[i]][j]].pt.y + _sadBlockSize + 1),
				cv::Range(nextCorner[_region[index1[i]][j]].pt.x - _sadBlockSize, nextCorner[_region[index1[i]][j]].pt.x + _sadBlockSize + 1)).copyTo(currPatch);
			double sad = cv::norm(prevPatch, currPatch, cv::NORM_L1);
			if (sad < minSAD)
			{
				minSAD = sad;
				//matchedCorner = nextCorner[_region[index1[i]][j]];
				matched.queryIdx = _region[index1[i]][j];
			}
		}
	}
	//穷尽一级搜索，取得SAD最小值后，再与指定阈值比较
	if (minSAD <= _sadThreshold1)
	{
		matches.push_back(matched);
		return; 
	}

	//二级搜索：周边16个区域
	size_t index2[16];
	index2[0] = yy*_regionXY[0] + xx + 2;
	index2[1] = (yy + 1)*_regionXY[0] + xx + 2;
	index2[2] = (yy + 2)*_regionXY[0] + xx + 2;
	index2[3] = (yy + 2)*_regionXY[0] + xx + 1;
	index2[4] = (yy + 2)*_regionXY[0] + xx;
	index2[5] = (yy + 2)*_regionXY[0] + xx - 1;
	index2[6] = (yy + 2)*_regionXY[0] + xx - 2;
	index2[7] = (yy + 1)*_regionXY[0] + xx - 2;
	index2[8] = yy*_regionXY[0] + xx - 2;
	index2[9] = (yy - 1)*_regionXY[0] + xx - 2;
	index2[10] = (yy - 2)*_regionXY[0] + xx - 2;
	index2[11] = (yy - 2)*_regionXY[0] + xx - 1;
	index2[12] = (yy - 2)*_regionXY[0] + xx;
	index2[13] = (yy - 2)*_regionXY[0] + xx + 1;
	index2[14] = (yy - 2)*_regionXY[0] + xx + 2;
	index2[15] = (yy - 1)*_regionXY[0] + xx + 2;
	for (int i = 0;i < 16;i++)
	{
		//对于处于边缘上的基准点进行额外处理（边缘上的基准点搜索区域没有这么多）
		if (xx - 1 < 0 && (i > 4 && i < 12)) { continue; }
		if (xx + 1 >= _regionXY[0] && (i < 4 || i>12)) { continue; }
		if (yy - 1 < 0 && i > 8) { continue; }
		if (yy + 1 >= _regionXY[1] && i < 8) { continue; }

		if (xx - 2 < 0 && (i > 5 && i < 11)) { continue; }
		if (xx + 2 == _regionXY[0] && (i < 3 || i>13)) { continue; }
		if (yy - 2 < 0 && (i > 9 && i < 15)) { continue; }
		if (yy + 2 == _regionXY[1] && (i > 1 && i < 7)) { continue; }

		pointsNum = _region[index2[i]].size();
		for (int j = 0;j < pointsNum;j++)
		{
			prev(cv::Range(nextCorner[_region[index2[i]][j]].pt.y - _sadBlockSize, nextCorner[_region[index2[i]][j]].pt.y + _sadBlockSize + 1),
				cv::Range(nextCorner[_region[index2[i]][j]].pt.x - _sadBlockSize, nextCorner[_region[index2[i]][j]].pt.x + _sadBlockSize + 1)).copyTo(currPatch);
			double sad = cv::norm(prevPatch, currPatch, cv::NORM_L1);
			if (sad < minSAD)
			{
				minSAD = sad;
				//matchedCorner = nextCorner[_region[index2[i]][j]];
				matched.queryIdx = _region[index2[i]][j];
			}
		}
	}
	//穷尽二级搜索，取得SAD最小值后，再与指定阈值比较
	if (minSAD <= _sadThreshold2)
	{
		matches.push_back(matched);
		return;
	}
}


void Matcher::calcSADPrevBased(
	cv::Mat prev, cv::Mat curr,
	cv::KeyPoint corner,
	size_t cornerID,
	std::vector<cv::KeyPoint> nextCorner,
	std::vector<cv::DMatch>& matches)
{
	int xx = (corner.pt.x - _originXY[0]) / _regionSize;
	int yy = (corner.pt.y - _originXY[1]) / _regionSize;

	//不考虑哪些不再区域内的基准点
	if (xx < 0 || xx >= _regionXY[0] || yy < 0 || yy >= _regionXY[1]) { return; }

	cv::Mat prevPatch, currPatch;
	cv::DMatch matched;

	//本匹配程序中，虽然根据两帧图像检测到的点的数目不同，分为以上一帧为基准点和当前帧为基准点，但这只是为了实现上的简单，
	//习惯上，还是以上一帧检测出来的角点为基准点，然后在当前帧中找对应点，因此，恒以上一帧中的点为queryIdx，当前帧中的点为trainIdx
	matched.queryIdx = cornerID;

	prev(cv::Range(corner.pt.y - _sadBlockSize, corner.pt.y + _sadBlockSize + 1),
		cv::Range(corner.pt.x - _sadBlockSize, corner.pt.x + _sadBlockSize + 1)).copyTo(prevPatch);

	double minSAD = _sadInitVal;
	//cv::KeyPoint matchedCorner;

	//零级搜索
	int index = yy*_regionXY[0] + xx;
	int pointsNum = _region[index].size();
	for (int i = 0;i < pointsNum;i++)
	{
		curr(cv::Range(nextCorner[_region[index][i]].pt.y - _sadBlockSize, nextCorner[_region[index][i]].pt.y + _sadBlockSize + 1),
			cv::Range(nextCorner[_region[index][i]].pt.x - _sadBlockSize, nextCorner[_region[index][i]].pt.x + _sadBlockSize + 1)).copyTo(currPatch);
		double sad = cv::norm(prev, curr, cv::NORM_L1);
		if (sad < minSAD)
		{
			minSAD = sad;
			//matchedCorner = nextCorner[_region[index][i]];
			matched.trainIdx = _region[index][i];
		}
	}
	//穷尽零级搜索，取得SAD最小值后，再与指定阈值比较
	if (minSAD <= _sadThreshold0)
	{
		matches.push_back(matched);
		return;			//小于0级SAD阈值，则视为匹配成功，存入匹配结果，返回，不再进行1级匹配
	}

	//一级搜索
	int index1[8];
	index1[0] = yy*_regionXY[0] + xx + 1;
	index1[1] = (yy + 1)*_regionXY[0] + xx + 1;
	index1[2] = (yy + 1)*_regionXY[0] + xx;
	index1[3] = (yy + 1)*_regionXY[0] + xx - 1;
	index1[4] = yy*_regionXY[0] + xx - 1;
	index1[5] = (yy - 1)*_regionXY[0] + xx - 1;
	index1[6] = (yy - 1)*_regionXY[0] + xx;
	index1[7] = (yy - 1)*_regionXY[0] + xx + 1;
	for (int i = 0;i < 8;i++)
	{
		//对于处于边缘上的基准点进行额外处理（边缘上的基准点搜索区域没有这么多）
		if (xx - 1 < 0 && (i > 2 && i < 6)) { continue; }
		if (xx + 1 == _regionXY[0] && (i == 0 || i == 1 || i == 7)) { continue; }
		if (yy - 1 < 0 && i > 4) { continue; }
		if (yy + 1 == _regionXY[1] && i < 4) { continue; }

		pointsNum = _region[index1[i]].size();
		for (int j = 0;j < pointsNum;j++)
		{
			curr(cv::Range(nextCorner[_region[index1[i]][j]].pt.y - _sadBlockSize, nextCorner[_region[index1[i]][j]].pt.y + _sadBlockSize + 1),
				cv::Range(nextCorner[_region[index1[i]][j]].pt.x - _sadBlockSize, nextCorner[_region[index1[i]][j]].pt.x + _sadBlockSize + 1)).copyTo(currPatch);
			double sad = cv::norm(prevPatch, currPatch, cv::NORM_L1);
			if (sad < minSAD)
			{
				minSAD = sad;
				//matchedCorner = nextCorner[_region[index1[i]][j]];
				matched.trainIdx = _region[index1[i]][j];
			}
		}
	}
	//穷尽一级搜索，取得SAD最小值后，再与指定阈值比较
	if (minSAD <= _sadThreshold1)
	{
		matches.push_back(matched);
		return;			//小于1级SAD阈值，则视为匹配成功，存入匹配结果，返回，不再进行2级匹配
	}

	//二级搜索
	size_t index2[16];
	index2[0] = yy*_regionXY[0] + xx + 2;
	index2[1] = (yy + 1)*_regionXY[0] + xx + 2;
	index2[2] = (yy + 2)*_regionXY[0] + xx + 2;
	index2[3] = (yy + 2)*_regionXY[0] + xx + 1;
	index2[4] = (yy + 2)*_regionXY[0] + xx;
	index2[5] = (yy + 2)*_regionXY[0] + xx - 1;
	index2[6] = (yy + 2)*_regionXY[0] + xx - 2;
	index2[7] = (yy + 1)*_regionXY[0] + xx - 2;
	index2[8] = yy*_regionXY[0] + xx - 2;
	index2[9] = (yy - 1)*_regionXY[0] + xx - 2;
	index2[10] = (yy - 2)*_regionXY[0] + xx - 2;
	index2[11] = (yy - 2)*_regionXY[0] + xx - 1;
	index2[12] = (yy - 2)*_regionXY[0] + xx;
	index2[13] = (yy - 2)*_regionXY[0] + xx + 1;
	index2[14] = (yy - 2)*_regionXY[0] + xx + 2;
	index2[15] = (yy - 1)*_regionXY[0] + xx + 2;
	for (int i = 0;i < 16;i++)
	{
		//对于处于边缘上的基准点进行额外处理（边缘上的基准点搜索区域没有这么多）
		if (xx - 1 < 0 && (i > 4 && i < 12)) { continue; }
		if (xx + 1 >= _regionXY[0] && (i < 4 || i>12)) { continue; }
		if (yy - 1 < 0 && i > 8) { continue; }
		if (yy + 1 >= _regionXY[1] && i < 8) { continue; }

		if (xx - 2 < 0 && (i > 5 && i < 11)) { continue; }
		if (xx + 2 == _regionXY[0] && (i < 3 || i>13)) { continue; }
		if (yy - 2 < 0 && (i > 9 && i < 15)) { continue; }
		if (yy + 2 == _regionXY[1] && (i > 1 && i < 7)) { continue; }

		pointsNum = _region[index2[i]].size();
		for (int j = 0;j < pointsNum;j++)
		{
			curr(cv::Range(nextCorner[_region[index2[i]][j]].pt.y - _sadBlockSize, nextCorner[_region[index2[i]][j]].pt.y + _sadBlockSize + 1),
				cv::Range(nextCorner[_region[index2[i]][j]].pt.x - _sadBlockSize, nextCorner[_region[index2[i]][j]].pt.x + _sadBlockSize + 1)).copyTo(currPatch);
			double sad = cv::norm(prevPatch, currPatch, cv::NORM_L1);
			if (sad < minSAD)
			{
				minSAD = sad;
				//matchedCorner = nextCorner[_region[index2[i]][j]];
				matched.trainIdx = _region[index2[i]][j];
			}
		}
	}
	//穷尽二级搜索，取得SAD最小值后，再与指定阈值比较
	if (minSAD <= _sadThreshold2)
	{
		matches.push_back(matched);
		return;
	}
}
