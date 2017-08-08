/*
* �ļ����ƣ�testMatcher.cpp
* �����⣺opencv2411�������汾
* ���Ի�����VS2015 x86
* ժ Ҫ������Matcherƥ����
* �����ļ���ReadImages.h/.cpp������ͼƬ�ࣻMatcher.h/.cppƥ���࣬������ȫ�����������Ǹò��Գ���Ĳ������ݣ�����ƥ��Ч����
* ˵���������Գ���ʹ�õ���FAST����������ȡ��ʵ����Matcherƥ���಻����ʹ�����������࣬��ΪMatcher��FASTû���κι�ϵ��
*		��ˣ�ֻҪ����ȡ��KeyPoint���������������ʹ�ø�Matcher����ƥ��
* �������ݣ��������ݰ������ף�һ����KITTI 01���ݼ�����10֡���˶����ȱȽϴ�_regionSizeȡ30����һ����ETH�����ݼ�����49֡���˶����Ƚ�С��_regionSizeȡ10��
			��������KITTI 02���ݼ�����101֡���˶�����Ҳ�Ƚϴ󣨵�ͼƬ��Щ���ֽ�С��������ȡ20��
* �����ѡ����ʵ�_regionSize�Խ��Ӱ�컹��ͦ��ģ������˶����Ƚ�С��������ڲ���̫��ȡһ����Сֵ������Ȼ����ɺܶ����ƥ�䣬���˷�ʱ�䣻�����˶����Ƚϴ�������
*		�����ʵ�ȡ�󣬵���Ҫ���󣬹��󲻻��������ɹ�ƥ��ĵ㣬��������������ƥ��㼰����ʱ��
* 2016.05.23
*/

#include <iostream>
#include<opencv2/opencv.hpp>
#include "Matcher.h"
#include "ReadImages.h"

int main()
{
	//����һ������ͼƬ��
	//ReadImages imgReader("G:\\KITTI_Database\\SLAM\\01", "000", ".png");
	//ReadImages imgReader("G:\\ETH ASL\\ETH-IMAGE-MODIFIED\\Img1", "", ".png");
	ReadImages imgReader("G:\\KITTI_Database\\SLAM\\02", "0000", ".png");


	//����һ��FAST���������
	cv::FastFeatureDetector fast(40);
	
	//�����洢fast�ǵ������
	std::vector<cv::KeyPoint> corner1;
	std::vector<cv::KeyPoint> corner2;

	//�����һ֡ͼ����һ֡��
	//cv::Mat prev = imgReader.loadImage(101, 0);
	//cv::Mat prev = imgReader.loadImage(1, 0);
	cv::Mat prev = imgReader.loadImage(1, 0);

	//�Ե�һ֡ͼ����fast�ǵ�
	fast.detect(prev, corner1);

	//��ǰ֡
	cv::Mat curr;

	//��֡�ı��ݣ����������滭�㣬��ʾ�����ͬʱ����ȾԭͼƬ
	cv::Mat prevCopy, currCopy;

	//�����洢ƥ�����ı���
	std::vector<cv::DMatch> matches;

	//����һ��fastƥ���࣬����ʼ����ÿ������Ĵ�СΪ30������SAD�Ŀ��СΪ2*2+1������������������Ϊ2��3��
	//Matcher matcher(prev.size(), 30, 2);
	//Matcher matcher(prev.size(), 10, 2);
	Matcher matcher(prev.size(), 20, 2);

	//��������ͼƬ����������֡��ƥ��
	//for (int i = 102;i < 111;i++)
	//for (int i = 2;i < 50;i++)
	for (int i = 2;i < 100;i++)
	{
		//������һ֡
		curr = imgReader.loadImage(i, 0);

		//����ͼ���Ա�����
		prev.copyTo(prevCopy);
		curr.copyTo(currCopy);

		//namedWindow("prev", 1);
		//namedWindow("curr", 1);
		//imshow("prev", prev);
		//imshow("curr", curr);
		//waitKey(2000);

		//fast�������
		fast.detect(curr, corner2);
		
		//ÿ��ƥ�䣬�ǵ������һ��ƥ��
		matches.clear();
		matcher.match(prev, curr, corner1, corner2, matches);

		//ƥ������ʾ
		size_t matchesNum = matches.size();
		std::cout << "���ڽ��е�" << i - 1 << "֡�͵�" << i << "֡ƥ�䣺" << std::endl;
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

		//��ǰ֡��⵽�Ľǵ㣬��ֵ����һ֡
		//corner1.clear();		//������仹�ǻ����
		//corner1 = corner2;
		//ʹ�������ֱ�Ӹ�ֵ����ʱ����ִ�����ʹ��ETH���ݼ������е���17֡ʱ��������ԭ������Expression:"(_Ptr_user & (_BIG_ALLOCATION_ALIGNMENT -1))==0" && 0
		//��������������������ڶ������Բ��ã�
		corner1.swap(corner2);
		corner2.clear();

		//��ǰ֡��ֵ����һ֡
		curr.copyTo(prev);
	}

	std::cout << "����֡��ɣ�" << std::endl;
	system("pause");
}