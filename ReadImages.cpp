/*
* �ļ����ƣ�ReadImages.cpp
* ժ Ҫ����ȡͼƬ�࣬ͼƬ���з��뵽һ���ļ����У���"C:\\imgFile"����һ����ʽ����������"0001.jpg","00010.jpg","1.jpg"
* ʹ��ʵ����
* ����ͷ�ļ���			#include "ReadImages.h"
* ������Ҫ������ͼƬ�ࣺReadImages imageReader("C:\\imgFile", "", ".jpg");
* ����ͼƬ��			Mat prev = imageReader.loadImage(1,0); //������"1.jpg"ͼƬ���Ҷȸ�ʽ
* 2016.05.23
*/

#include "ReadImages.h"
#pragma once

ReadImages::ReadImages(std::string basepath, const std::string imagename, const std::string suffix)
{
	//��·���е��ļ��зָ���ͳһΪ'/'�����Բ���Ҫ���forѭ�������Զ�ͳһ�������ʹ�õ�����ο�C++ Primer��
	for (auto &c : basepath)
	{
		if (c == '\\')
		{
			c = '/';
		}
	}

	_imgSource._basepath = basepath + "/";	//�������'/'��������'\\'
	_imgSource._imagename = imagename;		//ͼƬ����������ţ�
	_imgSource._suffix = suffix;				//ͼ����չ��

}

//���뵥��ͼƬ
//���룺imgId��һ��Ҫ��������ͼƬ��ͼƬ���и����
cv::Mat ReadImages::loadImage(int imgId, int imgType)
{	
	//��ͼƬ���ת���ַ���
	std::stringstream ss;
	std::string imgNum;
	ss << imgId;
	ss >> imgNum;

	//�õ�ͼƬ����������·��
	std::string path = _imgSource._basepath + _imgSource._imagename + imgNum + _imgSource._suffix;
	
	cv::Mat img = cv::imread(path,imgType);

	return img;
}