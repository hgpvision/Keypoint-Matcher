/*
* �ļ����ƣ�ReadImages.h
* ժ Ҫ����ȡͼƬ�࣬ͼƬ���з��뵽һ���ļ����У���"C:\\imgFile"����һ����ʽ����������"0001.jpg","00010.jpg","1.jpg"
* ʹ��ʵ����
* ����ͷ�ļ���			#include "ReadImages.h"
* ������Ҫ������ͼƬ�ࣺReadImages imageReader("C:\\imgFile", "", ".jpg");
* ����ͼƬ��			Mat prev = imageReader.loadImage(1,0); //������"1.jpg"ͼƬ���Ҷȸ�ʽ
* 2016.05.23
*/

#pragma once
#include <opencv2/opencv.hpp>

typedef struct ImgSource
{
	std::string _basepath = "";		//ͼƬ����·��
	std::string _imagename = "";	//ͼƬ����������ţ�
	std::string _suffix = "";		//ͼƬ��׺
}ImageSource;

class ReadImages
{
public:
	ReadImages() {}

	//@Brief��	��ȡͼƬ�๹�캯��
	//@Input��	basepath��ͼƬ�����ļ��еĻ���·������������\\��
	//			imagename��ͼƬ�����֣�������ţ�������ͼƬ�����ļ���"C:\\imgFile"�£�����Ϊ0001.jpg,0002.jpg,...,00010.jpg����imagenameΪ"000",1,2,10Ϊ���
	//			suffix��ͼƬ��׺�����ϣ���Ϊ".jpg"
	ReadImages(const std::string basepath, const std::string imagename, const std::string suffix);

	//@Brief��	����ͼƬ������imread()����
	//@Input��	imgId��ͼ����
	//			imgType������ͼƬ��ʽ����imread()ͳһ��0��ʾ����Ҷ�ͼ��1��ʾ�����ɫͼ
	//@Output��	��������ͼƬ��Mat��ʽ
	cv::Mat loadImage(int imgId, int imgType);	//����ͼƬ
private:
	ImageSource _imgSource;			//ͼƬ·����Ϣ
	cv::Rect _roi;
};
