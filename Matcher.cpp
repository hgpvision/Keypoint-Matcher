/*
* �ļ����ƣ�Matcher.cpp
* �����⣺opencv2411�������汾
* ���Ի�����VS2015 x86
* ժ Ҫ������SAD�����߶Ƚ���������ƥ�䣨SAD����������������ض�Ӧ�Ҷ�ֵ��ľ���ֵ�ĺͣ�
* ƥ����ԣ�������ͼƬ�ֳ�������ȥ�ܱ߲������أ�������ȥsadBlockSize�����أ���ֹ������������к�ѡ��������뵽��ͬ���ڣ�
*			����������������Ϊ���������Ƚ���0��ƥ�䣬���Ի�׼��������������һ֡�ж�Ӧ�����ڽ���ƥ�䣬ѡ��������е���СSADֵ����С��
*			�㼶SAD��ֵ������Ϊƥ��ɹ�������matches�����ٽ���1����2��ƥ�䣻1����2�����ƣ�����һ֡�У�����������������ƫ��1��2�������ܹ�8
*			��16�������ڽ���ƥ�䣩��1��ƥ���������1��SAD��ֵ�����ٽ���2��ƥ�䣻��2����������Ȼδ������ָ��2��SAD��ֵ��������õ㣬�Ӹõ�ƥ��ʧ�ܣ�
*			���������ƥ�䡣
* ˵������ƥ���࿪ʼ�����FAST����д�ģ���ʵ���ϸ�Matcher��FASTû���κι�ϵ����ˣ�ֻҪ����ȡ��KeyPoint���������������ʹ�ø�Matcher����ƥ��
* ʹ��ʵ������ʾ������testMatcher.cpp
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

	//��ȥһ������������������_regionSize�����⣬���뱣֤�������ܶ���_sadBlockSize����������ֹ�ڱ�Ե��������������Ե�ĵ��ڼ���SADֵʱ������ͼ��߽�
	size_t cols_croped = imgSize.width - 2*_sadBlockSize;
	size_t rows_croped = imgSize.height - 2*_sadBlockSize;

	//��ȡʣ��ͼ���ܹ��ֳɵ�������
	_regionXY[0] = cols_croped / _regionSize;
	_regionXY[1] = rows_croped / _regionSize;

	_totalRegionNum = _regionXY[0] * _regionXY[1];

	//��ΪҪ��ʣ��ͼ�����֣�����ʣ��ͼ�����ܿ��ܻ���һЩҪ���������أ�margin���հ�����
	marginX = cols_croped % _regionSize;
	marginY = rows_croped % _regionSize;

	//����ͼ����ߺ��ϱ��ܹ�����������������ȷ�����ռ�������ͼ���ԭ������
	_originXY[0] = marginX / 2 - 1 + _sadBlockSize;
	_originXY[1] = marginY / 2 - 1 + _sadBlockSize;

	//����ͼ���ұߺ��±��ܹ�����������������ȷ�����ռ�������ͼ����յ�����
	//_endXY[0] = cols - (marginX - marginX / 2 - 1) - _sadBlockSize;
	//_endXY[1] = rows - (marginY - marginX / 2 - 1) - _sadBlockSize;
	_endXY[0] = cols - (marginX - marginX / 2 )- 1 - _sadBlockSize;
	_endXY[1] = rows - (marginY - marginY / 2 )- 1 - _sadBlockSize;

	_prevORcurr = 1;

	_sadInitVal = 1000.0;

	//ѡ�ò�ͬ�Ŀ�ߴ磬����SAD��С����ߴ磬��ָ����SAD��ֵ��Ȼ��ͬ���������ظ�����ͬ������ƥ����ֻ���ǳߴ�Ϊ5��7��SAD�飨��Ӧ_sadBlockSizeΪ2��3��
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

	//�����һ֡��⵽�ĵ��٣������һ֡����������һ֡�еĵ���Ϊ��׼��
	if (prevCornerNum <= currCornerNum)
	{
		//�����Ҫ��������Ҫ��_region���³�ʼ��
		_region = std::vector<std::vector<size_t> >(_totalRegionNum);
		partition(currCorner, currCornerNum);
		_prevORcurr = 0;		//��Ϊ�����Ƕ���һ֡��������_prevORcurrΪ0

		for (int i = 0;i < prevCornerNum;i++)
		{
			calcSADPrevBased(prevImg, currImg, prevCorner[i], i, currCorner, matches);
		}
	}
	else
	{
		//�����һ֡��⵽�ĵ��٣������һ֡����������һ֡�еĵ���Ϊ��׼��
		//�����һ��ƥ�䣬�Ƕ���֡���з�������ô���ƥ��Ͳ����ٷ����ˣ�ֱ����֮ǰ�ķ�����������ǣ�����Ҫ���½��з���
		if (_prevORcurr)
		{
			//�����Ҫ��������Ҫ��_region���³�ʼ��
			_region = std::vector<std::vector<size_t> >(_totalRegionNum);
			partition(prevCorner, prevCornerNum);
		}
		_prevORcurr = 1;			//��Ϊ�Ƕ���һ֡��������Ϊ1
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
		//��margin���ĵ㲻�ڿ��ڣ���ȥ��Ҳ��Ϊ�˷�ֹ�������Ϊ����SADֵ���п��ܻᳬ��ͼ���Ե��������ȥһ�������أ�
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

	//��������Щ���������ڵĻ�׼��
	if (xx < 0 || xx >= _regionXY[0] || yy < 0 || yy >= _regionXY[1]) { return; }
	
	cv::Mat prevPatch, currPatch;
	cv::DMatch matched;

	//��ƥ������У���Ȼ������֡ͼ���⵽�ĵ����Ŀ��ͬ����Ϊ����һ֡Ϊ��׼��͵�ǰ֡Ϊ��׼�㣬����ֻ��Ϊ��ʵ���ϵļ򵥣�
	//ϰ���ϣ���������һ֡�������Ľǵ�Ϊ��׼�㣬Ȼ���ڵ�ǰ֡���Ҷ�Ӧ�㣬��ˣ�������һ֡�еĵ�ΪqueryIdx����ǰ֡�еĵ�ΪtrainIdx
	matched.trainIdx = cornerID;

	curr(cv::Range(corner.pt.y - _sadBlockSize, corner.pt.y + _sadBlockSize + 1),
		cv::Range(corner.pt.x - _sadBlockSize, corner.pt.x + _sadBlockSize + 1)).copyTo(prevPatch);

	double minSAD = _sadInitVal;
	//cv::KeyPoint matchedCorner;

	//�㼶����
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
	//�һ��������ȡ��SAD��Сֵ������ָ����ֵ�Ƚ�
	if (minSAD <= _sadThreshold0)
	{
		matches.push_back(matched);
		return;
	}

	//һ���������ܱ�8������
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
		//���ڴ��ڱ�Ե�ϵĻ�׼����ж��⴦����Ե�ϵĻ�׼����������û����ô�ࣩ
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
	//�һ��������ȡ��SAD��Сֵ������ָ����ֵ�Ƚ�
	if (minSAD <= _sadThreshold1)
	{
		matches.push_back(matched);
		return; 
	}

	//�����������ܱ�16������
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
		//���ڴ��ڱ�Ե�ϵĻ�׼����ж��⴦����Ե�ϵĻ�׼����������û����ô�ࣩ
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
	//�����������ȡ��SAD��Сֵ������ָ����ֵ�Ƚ�
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

	//��������Щ���������ڵĻ�׼��
	if (xx < 0 || xx >= _regionXY[0] || yy < 0 || yy >= _regionXY[1]) { return; }

	cv::Mat prevPatch, currPatch;
	cv::DMatch matched;

	//��ƥ������У���Ȼ������֡ͼ���⵽�ĵ����Ŀ��ͬ����Ϊ����һ֡Ϊ��׼��͵�ǰ֡Ϊ��׼�㣬����ֻ��Ϊ��ʵ���ϵļ򵥣�
	//ϰ���ϣ���������һ֡�������Ľǵ�Ϊ��׼�㣬Ȼ���ڵ�ǰ֡���Ҷ�Ӧ�㣬��ˣ�������һ֡�еĵ�ΪqueryIdx����ǰ֡�еĵ�ΪtrainIdx
	matched.queryIdx = cornerID;

	prev(cv::Range(corner.pt.y - _sadBlockSize, corner.pt.y + _sadBlockSize + 1),
		cv::Range(corner.pt.x - _sadBlockSize, corner.pt.x + _sadBlockSize + 1)).copyTo(prevPatch);

	double minSAD = _sadInitVal;
	//cv::KeyPoint matchedCorner;

	//�㼶����
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
	//��㼶������ȡ��SAD��Сֵ������ָ����ֵ�Ƚ�
	if (minSAD <= _sadThreshold0)
	{
		matches.push_back(matched);
		return;			//С��0��SAD��ֵ������Ϊƥ��ɹ�������ƥ���������أ����ٽ���1��ƥ��
	}

	//һ������
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
		//���ڴ��ڱ�Ե�ϵĻ�׼����ж��⴦����Ե�ϵĻ�׼����������û����ô�ࣩ
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
	//�һ��������ȡ��SAD��Сֵ������ָ����ֵ�Ƚ�
	if (minSAD <= _sadThreshold1)
	{
		matches.push_back(matched);
		return;			//С��1��SAD��ֵ������Ϊƥ��ɹ�������ƥ���������أ����ٽ���2��ƥ��
	}

	//��������
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
		//���ڴ��ڱ�Ե�ϵĻ�׼����ж��⴦����Ե�ϵĻ�׼����������û����ô�ࣩ
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
	//�����������ȡ��SAD��Сֵ������ָ����ֵ�Ƚ�
	if (minSAD <= _sadThreshold2)
	{
		matches.push_back(matched);
		return;
	}
}
