/*
* �ļ����ƣ�Matcher.h
* �����⣺opencv2411�������汾
* ���Ի�����VS2015 x86
* ժ Ҫ������SAD�����߶Ƚ���������ƥ�䡣
* ƥ����ԣ�������ͼƬ�ֳ�������ȥ�ܱ߲������أ�������ȥsadBlockSize�����أ���ֹ������������к�ѡ��������뵽��ͬ���ڣ�
*			����������������Ϊ���������Ƚ���0��ƥ�䣬���Ի�׼��������������һ֡�ж�Ӧ�����ڽ���ƥ�䣬ѡ��������е���СSADֵ����С��
*			�㼶SAD��ֵ������Ϊƥ��ɹ�������matches�����ٽ���1����2��ƥ�䣻1����2�����ƣ�����һ֡�У�����������������ƫ��1��2�������ܹ�8
*			��16�������ڽ���ƥ�䣩��1��ƥ���������1��SAD��ֵ�����ٽ���2��ƥ�䣻��2����������Ȼδ������ָ��2��SAD��ֵ��������õ㣬�Ӹõ�ƥ��ʧ�ܣ�
*			���������ƥ�䡣
* ˵������ƥ���࿪ʼ�����FAST����д�ģ���ʵ���ϸ�Matcher��FASTû���κι�ϵ����ˣ�ֻҪ����ȡ��KeyPoint���������������ʹ�ø�Matcher����ƥ��
* ʹ��ʵ������ʾ������testMatcher.cpp
* 2016.05.23
*/

#pragma once
#include <opencv2/opencv.hpp>

class Matcher
{
public:
	//@Brief�� FastMatcherƥ����Ĺ��캯��
	//@Input��	imgSize������ͼƬ�ĳߴ磬����ȷ������
	//			regionSize��ÿ������Ĵ�С�������α߳�
	//			sadBlockSize������SADֵ�Ŀ�Ĵ�С��ֻ��ȡ2��3����Ϊһ��ֵ��ʵ�ʴ�СΪ2*sadBlockSize+1��Ĭ��Ϊ2
	//@Output�� ��ֱ�������ֱ�Ӹı�������Ա���������һЩ�����������𵽳�ʼ����Ĳ���������
	Matcher(
		cv::Size imgSize, 
		int regionSize,
		int sadBlockSize =2);


	//@Brief��	ƥ�����������
	//@Input��	prevImg����һ֡ͼ��
	//			currImg����һ֡ͼ��
	//			prevCorner����һ֡ͼ���⵽������������
	//			currCorner����һ֡ͼ���⵽������������
	//			matches���洢ƥ����
	//@Output��	��ֱ�������ֱ�Ӹı����ò���matches��ֵ��ͬʱ�ı����Ա���_prevORcurr��ֵ
	void match(
		cv::Mat prevImg,
		cv::Mat currImg,
		std::vector<cv::KeyPoint> prevCorner,
		std::vector<cv::KeyPoint> currCorner,
		std::vector<cv::DMatch>& matches);


	//@Brief�� ���������������м�⵽�ĵ㣬���������С������ͼ����з�����������⵽����������뵽ָ����������
	//@Input��	corner��Ҫ���з���ͼƬ������⵽������������
	//			cornerNum����corner�ĳߴ磬������ĸ���
	//@Output�� ��ֱ����������յĽ��ֱ�Ӹı����Ա���_region
	void partition(std::vector<cv::KeyPoint> corner, size_t cornerNum);


	//@Brief��	�Ե�ǰ֡�е���������Ϊ��׼�㣬����һ֡��ƥ���Ӧ�㡣����SADֵ������ָ������ѡ������������һ֡�е�ƥ��㣨��ƥ���ϣ�
	//@Input��	prev����һ֡ͼ��
	//			curr����ǰ֡ͼ��
	//			corner��Ҫ����ƥ��Ļ�׼�㣨��ǰ֡�еĵ��������㣩
	//			nextCorner���˲�������ָ��һ֡������ָ��һ֡��������ָ��һ֡�����������㣬������Щ���������ҳ�corner��ƥ���
	//			matches��ƥ������queryIdx��ָ��һ֡ͼ���еĵ��ţ���prevCorner�е���������trainIdx��ָ��һ֡ͼ���еĵ��ţ���currCorner�е�������
	//@Output��	��ֱ�������ֱ�Ӹı�matches���ò�����ֵ
	void calcSADCurrBased(
		cv::Mat prev,
		cv::Mat curr,
		cv::KeyPoint corner, 
		size_t cornerID, 
		std::vector<cv::KeyPoint> nextCorner, 
		std::vector<cv::DMatch>& matches);


	//@Brief��	����һ֡�е���������Ϊ��׼�㣬�ڵ�ǰ֡��ƥ���Ӧ�㡣����SADֵ������ָ������ѡ������������һ֡�е�ƥ��㣨��ƥ���ϣ�
	//@Input��	prev����һ֡ͼ��
	//			curr����ǰ֡ͼ��
	//			corner��Ҫ����ƥ��Ļ�׼�㣨��һ֡�еĵ��������㣩
	//			nextCorner���˲�������ָ��һ֡������ָ��һ֡��������ָ��ǰ֡�����������㣬������Щ���������ҳ�corner��ƥ���
	//			matches��ƥ������queryIdx��ָ��һ֡ͼ���еĵ��ţ���prevCorner�е���������trainIdx��ָ��һ֡ͼ���еĵ��ţ���currCorner�е�������
	//@Output��	��ֱ�������ֱ�Ӹı�matches���ò�����ֵ
	void calcSADPrevBased(
		cv::Mat prev, 
		cv::Mat curr,
		cv::KeyPoint corner,
		size_t cornerID,
		std::vector<cv::KeyPoint> nextCorner,
		std::vector<cv::DMatch>& matches);
private:
	//SAD��ֵ���������ȼ���0����ƥ�伴�ڻ�׼������������ƥ�䣻1��ƥ�伴����������ƫ��1�������ܹ�8�������ڽ���ƥ�䣻2��ƫ�����������ܹ�16������
	double _sadThreshold0;		//0��ƥ��SAD��ֵ����С
	double _sadThreshold1;		//1��ƥ��SAD��ֵ���м�
	double _sadThreshold2;		//2��ƥ��SAD��ֵ�����
	double _sadInitVal;			//minSAD�ĳ�ʼֵ���ڹ��캯���н���ʼ��Ϊ1000

	//����SADֵ�Ŀ�Ĵ�С
	int _sadBlockSize;			//ʵ�ʴ�СΪ2*_sadBlockSize+1

	int _regionSize;			//ÿ��С����Ĵ�С
	int _totalRegionNum;		//����ͼƬ��ȥ�������غ����ֳ�������ܿ���������_regionXY[0]*_regionXY[1]
	size_t _originXY[2];		//���������ԭ�����꣨����ȥ�������غ��������ǲ���ͼ���ԭ�����꣩
	size_t _endXY[2];			//����������յ����꣨����ȥ�������غ��������ǲ���ͼ����յ����꣬���ε���һ���㣬��ԭ��ɶԽǹ�ϵ��
	size_t _regionXY[2];		//��x,y�������ֳ�����ĸ���

	//_region�����洢����������ñ�����һ����άvector����������_totalRegionNum��ʸ������Ӧÿ�����������Ȱ������У��е������������ֵ
	std::vector<std::vector<size_t> > _region;		//����ֳ������õ���ʸ�����飬�������е�ÿ��ʸ��������Ǳ����������������ֵ��ÿ��ʸ��Ԫ�ظ�������

	//���������־λ��Ϊ�˼�С�����Ĵ���������ǰһ��ƥ������Ƕ���֡���з��������´�ƥ���Ƕ���֡ͼ����з������򣬿��Բ����ٷ����ˣ�ֱ�������ϴ�ƥ������Ľ��
	bool _prevORcurr;								//��־λ����һ��ƥ���Ƕ���һ֡������1�����Ƕ���һ֡������0��
};