// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <iostream>
#include <stdio.h>
#include <queue>
#include <stack>
#include <random>


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}
/*
void rgb2hsv()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele de culoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_RGB2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}
*/
void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}

void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void lab1Ex2()
{
	Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img(i, j) = 255 - img(i, j);
		}
	}
	imshow("Negative image", img);
	waitKey(0);
}

void lab1Ex3(int val)
{
	Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img(i, j) += val;

			if (img(i, j) > 255)
				img(i, j) = 255;
			else
				if (img(i, j) < 0)
					img(i, j) = 0;
		}
	}
	imshow("Gray image (additive)", img);
	waitKey(0);
}

void lab1Ex4(int mul)
{
	Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img(i, j) = img(i, j) * mul; // additive factor
		}
	}
	imwrite("Images/newCameraman.bmp", img);
	imshow("Gray image (multiplicative)", img);
	waitKey(0);
}

void lab1Ex5()
{
	Mat_<Vec3b> img(256, 256, CV_8UC3);

	Vec3b white = Vec3b(255, 255, 255);
	Vec3b red = Vec3b(0, 0, 255);
	Vec3b green = Vec3b(0, 255, 0);
	Vec3b yellow = Vec3b(0, 255, 255);

	for (int i = 0; i < 128; i++)
		for (int j = 0; j < 128; j++)
			img(i, j) = white;

	for (int i = 0; i < 128; i++)
		for (int j = 128; j < 256; j++)
			img(i, j) = red;

	for (int i = 128; i < 256; i++)
		for (int j = 0; j < 128; j++)
			img(i, j) = green;

	for (int i = 128; i < 256; i++)
		for (int j = 128; j < 256; j++)
			img(i, j) = yellow;

	imshow("My new colored image", img);
	waitKey(0);
}

void lab2Ex1() {

	Mat_<Vec3b> src = imread("Images/flowers_24bits.bmp", 1);

	int height = src.rows;
	int width = src.cols;

	Mat_<uchar> dstB(height, width);
	Mat_<uchar> dstG(height, width);
	Mat_<uchar> dstR(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			Vec3b pixel = src(i, j);

			uchar B = pixel[0];
			uchar G = pixel[1];
			uchar R = pixel[2];

			dstB(i, j) = B;
			dstG(i, j) = G;
			dstR(i, j) = R;
		}
	}

	imshow("Input image", src);
	imshow("B", dstB);
	imshow("G", dstG);
	imshow("R", dstR);
	waitKey(0);
}

void lab2Ex2() {

	Mat_<Vec3b> src = imread("Images/flowers_24bits.bmp", 1);

	int height = src.rows;
	int width = src.cols;

	Mat_<uchar> dst(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			Vec3b pixel = src(i, j);

			uchar B = pixel[0];
			uchar G = pixel[1];
			uchar R = pixel[2];

			dst(i, j) = (B + G + R) / 3;
		}
	}

	imshow("Input image", src);
	imshow("Grayscale image", dst);
	waitKey(0);
}

void lab2Ex3() {

	Mat_<uchar> src = imread("Images/cameraman.bmp", 0);
	imshow("Input image", src); // intial image
	int threshold;
	printf("Enter value: ");
	scanf("%d", &threshold);
	int height = src.rows;
	int width = src.cols;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			if (src(i, j) >= threshold)
				src(i, j) = 255;	// WHITE
			else
				if (src(i, j) < threshold)
					src(i, j) = 0;	// BLACK
		}
	}

	imshow("Black&White image", src);
	waitKey(0);
	waitKey(0);
}

float compute_saturation(float V, float C) {
	float S;
	if (V != 0)
		S = C / V;
	else
		S = 0;

	return S;
}

float compute_hue(float C, float M, float r, float g, float b) {
	float H;

	if (C != 0) {
		if (M == r)
			H = 60 * (g - b) / C;
		else if (M == g)
			H = 120 + 60 * (b - r) / C;
		else if (M == b)
			H = 240 + 60 * (r - g) / C;
	}
	else
		H = 0;	// grayscale

	if (H < 0)
		H = H + 360;

	return H;
}

Mat lab2Ex4() {

	Mat_<Vec3b> src = imread("Images/Lena_24bits.bmp", 1);

	int height = src.rows;
	int width = src.cols;

	Mat_<uchar> dstH(height, width);
	Mat_<uchar> dstS(height, width);
	Mat_<uchar> dstV(height, width);

	Mat_<Vec3b> dstHSV(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			Vec3b pixel = src(i, j);

			float B = pixel[0];
			float G = pixel[1];
			float R = pixel[2];

			float b = B / 255; // b : the normalized B component 
			float g = G / 255; // g : the normalized G component
			float r = R / 255; // r : the normalized R component

			float M = max(r, max(g, b));	// V = M
			float m = min(r, min(g, b));
			float C = M - m;

			float V = M;
			float S = compute_saturation(V, C);
			float H = compute_hue(C, M, r, g, b);

			dstV(i, j) = V * 255;
			dstS(i, j) = S * 255;
			dstH(i, j) = H * 255 / 360;

			dstHSV(i, j)[0] = H / 2;
			dstHSV(i, j)[1] = S * 255;
			dstHSV(i, j)[2] = V * 255;
		}
	}

	imshow("Initial image", src);
	imshow("H", dstH);
	imshow("S", dstS);
	imshow("V", dstV);
	imshow("HSV", dstHSV);
	waitKey(0);

	return dstHSV;
}

bool isInside(Mat img, int i, int j) {
	int height = img.rows;
	int width = img.cols;

	if (i >= 0 && i < height) {
		if (j >= 0 && j < width) {
			return true;
		}
		else {
			return false;
		}
	}
	else {
		return false;
	}
}

void lab3Ex1(Mat_<uchar> img, int* histogram) {

	for (int i = 0; i < 256; i++) {
		histogram[i] = 0;
	}

	for (int j = 0; j < img.rows; j++) {
		for (int k = 0; k < img.cols; k++) {
			histogram[img(j, k)]++;
		}
	}
}

void lab3Ex2(Mat_<uchar>img, float* pdf, int* histogram)
{
	lab3Ex1(img, histogram);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			uchar pixel = img(i, j);
			pdf[pixel] = float(histogram[pixel]) / float(img.rows * img.cols);
		}
	}
}

void lab3Ex3(int* histogram, int height, int nr_bins) {

	Mat imgHist(height, nr_bins, CV_8UC3, CV_RGB(255, 255, 255));

	int max_hist = 0;
	for (int i = 0; i < nr_bins; i++)
		if (histogram[i] > max_hist)
			max_hist = histogram[i];

	double scale = 1.0;
	scale = (double)height / max_hist;
	int baseline = height - 1;

	for (int j = 0; j < nr_bins; j++) {
		for (int k = baseline; k > baseline - histogram[j] * scale; --k) {
			imgHist.at<Vec3b>(k, j) = { 0, 255, 0 };
		}
	}

	imshow("Histogram", imgHist);
	waitKey(0);
}

void lab3Ex4(Mat_<uchar> img, int* histogram, int nr_bins)
{

	for (int i = 0; i < 256; i++) {
		histogram[i] = 0;
	}

	for (int j = 0; j < img.rows; j++) {
		for (int k = 0; k < img.cols; k++) {
			histogram[(int)img(j, k) / (256 / nr_bins)]++;
		}
	}
}

std::vector<int> getMaxim(Mat_<uchar> img, int wh, float th) {

	int hist[256];
	float pdf[256];
	lab3Ex1(img, hist);
	lab3Ex2(img, pdf, hist);

	std::vector<int>maxima;
	maxima.push_back(0);

	for (int k = 0 + wh; k <= 255 - wh; k++) {
		float v = 0;
		for (int i = k - wh; i <= k + wh; i++) {
			v += pdf[i];
		}
		v = v / (2 * wh + 1);
		bool ok = true;
		if (pdf[k] > v + th) {
			for (int i = k - wh; i <= k + wh; i++) {
				if (pdf[k] < pdf[i]) {
					ok = false;
				}
			}
		}
		else
			ok = false;
		if (ok == true) {
			maxima.push_back(k);
		}
	}
	maxima.push_back(255);
	return maxima;
}

uchar findClosestMax(int pixel, std::vector<int> max) {
	uchar closest = 0;
	int differe = 256;
	for (int i : max) {
		if (abs(pixel - i) <= differe) {
			differe = abs(pixel - i);
			closest = i;
		}
	}
	return closest;
}

void lab3Ex5(Mat_<uchar> img, int wh, float th) {

	std::vector<int> maximumPositions = getMaxim(img, wh, th);

	int height = img.rows;
	int width = img.cols;
	Mat_<uchar> destination(height, width);
	int vect[256];
	for (int i = 0; i <= 255; i++)
	{
		vect[i] = findClosestMax(i, maximumPositions);
	}
	for (int i = 0; i <= 255; i++)
	{
		printf("%d ", vect[i]);
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			destination(i, j) = vect[img(i, j)];
		}
	}
	imshow("Tresholding image", destination);
	waitKey(0);
}

void multiLevelTh(Mat_<uchar> img, int* hist)
{
	lab3Ex1(img, hist);
	float pdf[256];
	lab3Ex2(img, pdf, hist);
	int WH = 5;
	float TH = 0.0003;
	int maxima[256];
	maxima[0] = 0;
	int index = 1;
	float v, maxValue;
	for (int i = 0 + WH; i <= 255 - WH; i++)
	{
		v = 0.0;
		maxValue = 0.0;
		for (int j = i - WH; j <= i + WH; j++)
		{
			v += pdf[j];
			maxValue = max(maxValue, pdf[j]);
		}

		v = v / (2 * WH + 1);
		if (pdf[i] > v + TH && pdf[i] >= maxValue)
		{
			maxima[index] = i;
			index++;
		}
	}
	maxima[index] = 255;
	for (int i = 0; i <= index; i++)
	{
		std::cout << maxima[i] << std::endl;
	}



}

void lab3Ex6(Mat_<uchar> img) {

	std::vector<int> maximumPositions = getMaxim(img, 5, 0.0003);

	Mat_<uchar> dst(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			dst(i, j) = img(i, j);
		}
	}

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			int old_pixel = dst(i, j);
			int new_pixel = findClosestMax(old_pixel, maximumPositions);
			dst(i, j) = new_pixel;
			int err = old_pixel - new_pixel;

			if (isInside(img, i, j + 1)) {
				dst(i, j + 1) = img(i, j + 1) + 7 * err / 16;
			}
			else if (isInside(img, i + 1, j - 1)) {
				dst(i + 1, j - 1) = img(i + 1, j - 1) + 3 * err / 16;
			}
			else if (isInside(img, i + 1, j)) {
				dst(i + 1, j) = img(i + 1, j) + 5 * err / 16;
			}
			else if (isInside(img, i + 1, j + 1)) {
				dst(i + 1, j + 1) = img(i + 1, j + 1) + err / 16;
			}
		}
	}
	imshow("Initial image", img);
	imshow("New image", dst);
	waitKey(0);
}

Mat_<Vec3b> lab4Ex2_GenerateColors(int height, int width, Mat_<int> labels);

void lab4Ex1_BFS() {

	Mat_<uchar> img = imread("Images/lab4/diagonal.bmp", IMREAD_GRAYSCALE);
	int height = img.rows;
	int width = img.cols;

	Mat_<int> labels(height, width);
	labels.setTo(0); 

	int di[4] = { -1, 0, 1, 0 };
	int dj[4] = { 0, -1, 0, 1 };

	int label = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			if (img(i, j) == 0 && labels(i, j) == 0) {

				label++;
				std::queue<Point2i> Q;
				labels(i, j) = label;
				Q.push(Point2i(j, i));

				while (!Q.empty()) {

					Point2i q = Q.front();
					Q.pop();


					for (int k = 0; k < 4; k++) {

						if (isInside(img, q.y + di[k], q.x + dj[k])) {
							uchar neigh = img(q.y + di[k], q.x + dj[k]);

							if (neigh == 0 && labels(q.y + di[k], q.x + dj[k]) == 0) {
								labels(q.y + di[k], q.x + dj[k]) = label;
								Q.push(Point2i(q.x + dj[k], q.y + di[k]));
							}
						}
					}
				}
			}
		}
	}

	Mat_<Vec3b> coloredImg = lab4Ex2_GenerateColors(height, width, labels);

	imshow("Initial image", img);
	imshow("BFS", coloredImg);
	waitKey(0);
}

Mat_<Vec3b> lab4Ex2_GenerateColors(int height, int width, Mat_<int> labels) {
	Mat_<Vec3b> dst(height, width);

	int maxLabel = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (labels(i, j) > maxLabel) {
				maxLabel = labels(i, j);
			}
		}
	}

	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);

	std::vector<Vec3b> colors(maxLabel + 1);
	for (int i = 0; i <= maxLabel; i++) {

		uchar r = d(gen);
		uchar g = d(gen);
		uchar b = d(gen);

		colors.at(i) = Vec3b(r, g, b);
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			int label = labels(i, j);

			if (label > 0)
				dst(i, j) = colors.at(labels(i, j));
			else
				dst(i, j) = Vec3b(255, 255, 255);
		}
	}

	return dst;
}

void lab4Ex3() {
	Mat_<uchar> img = imread("Images/lab4/diagonal.bmp", IMREAD_GRAYSCALE);
	int height = img.rows;
	int width = img.cols;

	Mat_<int> labels(height, width);
	labels.setTo(0); 

	int label = 0;

	int di[4] = { 0, -1, -1, -1 };
	int dj[4] = { -1, -1, 0, 1 };

	std::vector<std::vector<int>> edges(1000);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (img(i, j) == 0 && labels(i, j) == 0) {
				std::vector<int> L;
				for (int k = 0; k < 4; k++) {
					if (isInside(img, i + di[k], j + dj[k])) {
						if (labels(i + di[k], j + dj[k]) > 0) {
							L.push_back(labels(i + di[k], j + dj[k]));
						}
					}
				}
				if (L.size() == 0) {
					label++;
					labels(i, j) = label;
				}
				else {
					int x = *std::min_element(L.begin(), L.end());
					labels(i, j) = x;
					for (int y : L) {
						if (y != x) {
							edges[x].push_back(y);
							edges[y].push_back(x);
						}
					}
				}
			}
		}
	}

	Mat_<Vec3b> firstImg = lab4Ex2_GenerateColors(height, width, labels);
	imshow("Initial image", img);
	imshow("First pass", firstImg);

	int newLabel = 0;
	int* newLabels = new int[label + 1];
	for (int i = 0; i <= label; i++) {
		newLabels[i] = 0;
	}

	for (int j = 1; j <= label; j++) {
		if (newLabels[j] == 0) {
			newLabel++;
			std::queue<int> Q;
			newLabels[j] = newLabel;
			Q.push(j);

			while (!Q.empty()) {
				int x = Q.front();
				Q.pop();
				for (int y : edges[x]) {
					if (newLabels[y] == 0) {
						newLabels[y] = newLabel;
						Q.push(y);
					}
				}
			}
		}
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			labels(i, j) = newLabels[labels(i, j)];
		}
	}


	Mat_<Vec3b> secondImg = lab4Ex2_GenerateColors(height, width, labels);

	imshow("Second pass", secondImg);
	waitKey(0);

}

void lab5Ex1() {
	Mat_<uchar> img = imread("Images/lab5/so/triangle_up.bmp", IMREAD_GRAYSCALE);

	Mat_<Vec3b> dst(img.rows, img.cols);

	double area = 0;

	double row_center = 0;
	double col_center = 0;

	double numarator = 0;
	double numitor = 0;

	double perimeter = 0;
	double T;

	double R;

	// blue
	Vec3b blue;
	blue.val[0] = 255;
	blue.val[1] = 0;
	blue.val[2] = 0;

	// WHITE
	Vec3b white;
	white.val[0] = 255;
	white.val[1] = 255;
	white.val[2] = 255;

	// BLACK
	Vec3b black;
	black.val[0] = 0;
	black.val[1] = 0;
	black.val[2] = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0) {
				area++; // calculate area

				dst(i, j) = blue; // color the figure in blue

				row_center += i;
				col_center += j;
			}
			else {
				dst(i, j) = white;
			}
		}
	}

	row_center = (double)(row_center / area);
	col_center = (double)(col_center / area);

	// cross at the center of mass
	for (int k = 0; k < 5; k++) {
		dst(row_center + k, col_center) = black;
		dst(row_center - k, col_center) = black;
		dst(row_center, col_center + k) = black;
		dst(row_center, col_center - k) = black;
	}

	// axis of elongation
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0) {
				numarator += 2 * (i - row_center) * (j - col_center);
				numitor += (((j - col_center) * (j - col_center)) - ((i - row_center)*(i - row_center)));
			}
		}
	}

	double phi = atan2(numarator, numitor);
	phi = phi / 2;
	double phi1 = phi * 180 / PI;

	// draw elongation axis
	line(dst, Point(col_center + 200 * cos(phi), row_center + 200 * sin(phi)), Point(col_center - 200 * cos(phi), row_center - 200 * sin(phi)), Scalar(0, 0, 255));

	// Draw perimeter
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0) {
				if (img(i + 1, j) != 0 || img(i - 1, j) != 0 || img(i, j + 1) != 0 || img(i, j - 1) != 0) {
					perimeter++;
					dst(i, j) = black;
				}
			}
		}
	}

	// thinness ratio
	T = 4 * PI * (area / (perimeter*perimeter));

	// the aspect ratio
	int c_max = 0;
	int c_min = img.cols;
	int r_max = 0;
	int r_min = img.rows;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0) {
				if (j > c_max)
					c_max = j;
				if (j < c_min)
					c_min = j;
				if (i > r_max)
					r_max = i;
				if (i < r_min)
					r_min = i;
			}
		}
	}

	R = (c_max - c_min + 1) / (r_max - r_min + 1);
	line(dst, Point(c_min, r_min), Point(c_min, r_max), Scalar(0, 0, 0));
	line(dst, Point(c_min, r_max), Point(c_max, r_max), Scalar(0, 0, 0));
	line(dst, Point(c_max, r_max), Point(c_max, r_min), Scalar(0, 0, 0));
	line(dst, Point(c_max, r_min), Point(c_min, r_min), Scalar(0, 0, 0));

	// projections of the binary object
	int* h = (int*)malloc(img.rows*sizeof(int)); // horizontal
	int* v = (int*)malloc(img.cols*sizeof(int)); // vertical

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0) {
				h[i]++;
				v[j]++;
			}
		}

	for (int i = 0; i < img.rows; i++) {
		line(dst, Point(0, i), Point(h[i], i), Scalar(0, 0, 0));
	}
	for (int j = 0; j < img.cols; j++) {
		line(dst, Point(j, 0), Point(j, v[j]), Scalar(0, 0, 0));
	}

	printf("Area = %lf\n", area);

	printf("Center of mass: row = %lf, col = %lf\n", row_center, col_center);

	printf("Angle: in radians = %fl, in degrees = %fl\n", phi, phi1);

	printf("The thinness ratio T = %lf\n", T);

	imshow("Before", img);
	imshow("After", dst);

	waitKey(0);
}

void lab6_borderTracing()
{
	std::vector<int> chainCode;
	Mat_<uchar> src = imread("Images/lab5/so/star_Z125%.bmp", IMREAD_GRAYSCALE);
	bool finnish = false;
	int height = src.rows;
	int width = src.cols;

	//N8
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	Point2i borderPoints[10000];
	
	int counter = 0;
	Point2i point;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (src(i, j) == 0)
			{
				point = Point2i(j, i);
				borderPoints[counter] = point;
				counter++;
				finnish = true;
				break;
			}
		}
		if (finnish)
			break;
	}

	int dir = 7;
	while (!(borderPoints[0] == borderPoints[counter - 2] && borderPoints[1] == borderPoints[counter - 1]) || (counter <= 2)) 
	{
		if (dir % 2 == 1)// odd
			dir = (dir + 6) % 8;
		else //even
			dir = (dir + 7) % 8;

		for (int k = dir; k < dir + 8; k++) 
		{
			uchar neighbor = src(point.y + di[k % 8], point.x + dj[k % 8]);
			if (neighbor == 0) {
				point = Point2i(point.x + dj[k % 8], point.y + di[k % 8]);
				borderPoints[counter] = point;

				chainCode.push_back(dir);
				dir = k % 8;

				counter++;
				break;
			}
		}
	}

	Mat_<uchar> dst(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst(i, j) = 0;
		}
	}

	for (int i = 0; i < counter; i++) {
		dst(borderPoints[i].y, borderPoints[i].x) = 255;
	}

	//chain codes
	printf("Chain code\n");
	for (int i = 0; i < chainCode.size(); i++) {
		printf("%d ", chainCode[i]);
	}

	printf("\nDerivative chain code\n");
	for (int i = 1; i < chainCode.size(); i++) {
		int deriv = (chainCode[i] - chainCode[i - 1] + 8) % 8;
		printf("%d ", deriv);
	}

	imshow("Initial image", src);
	imshow("Bordered", dst);
	waitKey(0);
}

bool isInside2(Mat img, int i, int j) {

	int height = img.rows;
	int width = img.cols;

	if (i >= 0 && i < height) {
		if (j >= 0 && j < width) {
			return true;
		}
		else {
			return false;
		}
	}
	else {
		return false;
	}
}

Mat_<uchar> generate_StructElem(int n) {

	Mat_<uchar> struct_elem(n, n, 255);
	int middle = n / 2;
	for (int i = 0; i < n; i++) {
		struct_elem(i, middle) = 0;
		struct_elem(middle, i) = 0;
	}
	return struct_elem;
}

Mat_<uchar> dilation(Mat_<uchar> src, Mat_<uchar> struct_elem) {
	int height = src.rows;
	int width = src.cols;
	Mat_<uchar> dst = src.clone();

	int structElem_i = struct_elem.rows / 2;
	int structElem_j = struct_elem.cols / 2;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src(i, j) == 0) {

				dst(i, j) = 0;

				for (int a = 0; a < struct_elem.rows; a++) {
					for (int b = 0; b < struct_elem.cols; b++) {
						if (struct_elem(a, b) == 0) {
							if (isInside2(src, i + a - structElem_i, j + b - structElem_j)) {
								dst(i + a - structElem_i, j + b - structElem_j) = 0;
							}
						}
					}
				}
			}
		}
	}
	return dst;
}

Mat_<uchar> erosion(Mat_<uchar> src, Mat_<uchar> struct_elem) {
	int height = src.rows;
	int width = src.cols;
	Mat_<uchar> dst = src.clone();

	int structElem_i = struct_elem.rows / 2;
	int structElem_j = struct_elem.cols / 2;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src(i, j) == 0) {

				for (int a = 0; a < struct_elem.rows; a++) {
					for (int b = 0; b < struct_elem.cols; b++) {
						if (struct_elem(a, b) == 0) {
							if (isInside2(src, i + a - structElem_i, j + b - structElem_j)) {
								if (src(i + a - structElem_i, j + b - structElem_j) == 255) {
									dst(i, j) = 255;
								}
							}
						}
					}
				}
			}
		}
	}
	return dst;
}

void opening(Mat_<uchar> src, Mat_<uchar> elem)
{
	imshow("Initial image", src);
	
	Mat_<uchar> dst(src.rows / 2, src.cols / 2);

	dst = dilation(erosion(src, elem), elem);

	imshow("Opening", dst);
	waitKey(0);
}

void closing(Mat_<uchar> src, Mat_<uchar> elem)
{
	imshow("Initial image", src);

	Mat_<uchar> dst(src.rows / 2, src.cols / 2);

	dst = erosion(dilation(src, elem), elem);

	imshow("Closing", dst);
	waitKey(0);
}

Mat_<uchar> boundaryExtraction(Mat_<uchar> src, Mat_<uchar> elem) {

	int height = src.rows;
	int width = src.cols;
	Mat_<uchar> dst = src.clone();

	Mat_<uchar> afterErosion = erosion(src, elem);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src(i, j) == afterErosion(i, j)) {
				dst(i, j) = 255;
			}
			else {
				dst(i, j) = 0;
			}
		}
	}

	return dst;
}

bool compareIterations(Mat_<uchar> mat1, Mat_<uchar> mat2) {
	for (int i = 0; i < mat1.rows; i++) {
		for (int j = 0; j < mat1.cols; j++) {
			if (mat1(i, j) != mat2(i, j)) {
				return false;
			}
		}
	}
	return true;
}

Mat_<uchar> regionFilling(Mat_<uchar> src, Mat_<uchar> structElem) {
	std::pair<int, int> point;
	bool stop = false;
	//find the starting pixel
	for (int i = 0; i < src.rows; i++) {
		int pass = 0;
		for (int j = 0; j < src.cols; j++) {
			if (pass == 0 && src(i, j) == 0) {
				pass = 1;
			}
			if (pass == 1 && src(i, j) == 255) {
				point = { i, j };
				pass = 2;
			}
			if (pass == 2 && src(i, j) == 0) {
				pass = 3;
			}
			if (pass == 3 && src(i, j) == 255) {
				stop = true;
				break;
			}
		}
		if (stop == true) {
			break;
		}
	}

	Mat_<uchar> inverse(src.rows, src.cols);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src(i, j) == 0) {
				inverse(i, j) = 255;
			}
			else inverse(i, j) = 0;
		}
	}

	Mat_<uchar> mat(src.rows, src.cols, 255);
	mat(point.first, point.second) = 0;

	Mat_<uchar> mat1(src.rows, src.cols, 255);

	while (1) {
		Mat_<uchar> aux = dilation(mat, structElem);
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (aux(i, j) == 0 && inverse(i, j) == 0) {
					mat1(i, j) = 0;
				}
				else {
					mat1(i, j) = 255;
				}
			}
		}
		//imshow("Fill", mat);
		if (compareIterations(mat, mat1)) {
			break;
		}
		mat = mat1.clone();
	}
	return mat;

}

void computeHistogram1(Mat_<uchar> img, int* hist, int nr_bins) {

	for (int i = 0; i < 256; i++) {
		hist[i] = 0;
	}

	for (int j = 0; j < img.rows; j++) {
		for (int k = 0; k < img.cols; k++) {
			hist[img(j, k)]++; 
		}
	}
}

void showHistogram1(int* hist, int height, int nr_bins) {
	Mat imgHist(height, nr_bins, CV_8UC3, CV_RGB(255, 255, 255)); // white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < nr_bins; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];

	double scale = 1.0;
	scale = (double)height / max_hist;
	int baseline = height - 1;

	for (int j = 0; j < nr_bins; j++) {
		for (int k = baseline; k > baseline - hist[j] * scale; --k) {
			imgHist.at<Vec3b>(k, j) = { 0, 255, 0 };
		}
	}
	imshow("Histogram", imgHist);
	waitKey(0);
}

float meanValue(Mat_<uchar> img, int hist[256]) {
	int M = img.rows * img.cols;
	float g = 0;

	for (int i = 0; i < 256; i++) {
		g += i * hist[i];
	}
	g = (float)g / M;
	return g;
}

float standardDeviation(Mat_<uchar> img, float mean, int hist[256]) {

	float dev = 0;
	int M = img.rows * img.cols;

	for (int g = 0; g < 256; g++) {
		dev += (float)(g - mean) * (g - mean) * hist[g];
	}
	dev = (float)dev / M;
	dev = sqrt(dev);
	return dev;
}

Mat_<uchar> grayscaleToBW1(Mat_<uchar> src, int treshold) {
	int height = src.rows;
	int width = src.cols;
	Mat_<uchar> dst(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			if (src(i, j) >= treshold)
				dst(i, j) = 255;
			else
				if (src(i, j) < treshold)
					dst(i, j) = 0;
		}
	}

	imshow("Initial", src);
	imshow("Black&White image", dst);
	waitKey(0);

	return dst;
}

void lab8_globalTresholding(Mat_<uchar> img, int hist[256]) {

	int height = img.rows;
	int width = img.cols;

	int iMin = hist[0];
	int iMax = hist[0];

	Mat_<uchar> dst(height, width);

	for (int i = 0; i < 256; i++) {
		if (hist[i] != 0 && i < iMin) {
			iMin = i;
		}
		if (hist[i] != 0 && i > iMax) {
			iMax = i;
		}
	}


	float T = (float)(iMin + iMax) / 2;
	float T_previous = T;
	do {
		float meanG1 = 0;
		float meanG2 = 0;

		float N1 = 0;
		float N2 = 0;

		for (int i = iMin; i <= T; i++) {

			meanG1 += i * hist[i];
			N1 += hist[i];
		}
		meanG1 = meanG1 / N1;

		for (int j = T + 1; j <= iMax; j++) {

			meanG2 += j * hist[j];
			N2 += hist[j];
		}
		meanG2 = meanG2 / N2;

		T_previous = T;
		T = (float)(meanG1 + meanG2) / 2;
	} while (abs(T - T_previous) > 0);

	dst = grayscaleToBW1(img, T);
}

void contrastChange(Mat_<uchar> img, int g_outMin, int g_outMax, int hist[256]) {

	int height = img.rows;
	int width = img.cols;

	Mat_<uchar> dst(height, width);

	int g_inMax = img(0,0);
	int g_inMin = img(0,0);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			g_inMax = max(img(i, j),g_inMax);
			g_inMin = min(g_inMin,img(i, j));
		}
	}

	float raport = (float)(g_outMax - g_outMin) / (g_inMax - g_inMin);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			int g_in = img(i, j);
			int g_out = g_outMin + (g_in - g_inMin) * raport;

			if (g_out <= 0) 
				dst(i, j) = 0;
			else if (g_out >= 255)
				dst(i, j) = 255;
			else
				dst(i, j) = g_out;
		}
	}

	imshow("Initial image", img);
	imshow("After", dst);
	waitKey(0);
}

void gamma_correction(Mat_<uchar> img, float gamma0, float gamma1) {

	int height = img.rows;
	int width = img.cols;

	Mat_<uchar> dst0(height, width); // compression
	Mat_<uchar> dst1(height, width); // decompression


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			float val0 = 255 * (pow(((float)img(i, j) / 255), gamma0));
			float val1 = 255 * (pow(((float)img(i, j) / 255), gamma1));

			if (val0 >= 255)
				dst0(i, j) = 255;
			else if (val0 <= 0)
				dst0(i, j) = 0;
			else dst0(i, j) = val0;

			if (val1 >= 255)
				dst1(i, j) = 255;
			else if (val1 <= 0)
				dst1(i, j) = 0;
			else dst1(i, j) = val1;
		}
	}

	imshow("Initial", img);
	imshow("Compression", dst0);
	imshow("Decompression", dst1);
	waitKey(0);

}

Mat_<float> convolution(Mat_<uchar> img, Mat_<float> H) {

	int height = img.rows;
	int width = img.cols;

	int w_rows = H.rows;
	int w_cols = H.cols;

	Mat_<float> res(height, width);

	int k_row = (H.rows - 1) / 2;
	int k_col = (H.cols - 1) / 2;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			float sum = 0;

			for (int u = 0; u < w_rows; u++) {
				for (int v = 0; v < w_cols; v++) {

					if (isInside(img, i + u - k_row, j + v - k_col)) {

						sum += H(u, v) * img(i + u - k_row, j + v - k_col);
					}
				}
			}

			res(i, j) = sum;
		}
	}

	return res;
}

Mat_<uchar> normalization(Mat_<float> H, Mat_<float> convo) {
	int height = convo.rows;
	int width = convo.cols;

	Mat_<uchar> dst(height, width);

	int w_rows = H.rows;
	int w_cols = H.cols;

	float sum_negative = 0.0;
	float sum_positive = 0.0;

	bool allPos = true;
	for (int i = 0; i < w_rows; i++) {
		for (int j = 0; j < w_cols; j++) {

			if (H(i, j) > 0) {
				sum_positive += H(i, j);
			}
			else if (H(i, j) < 0) {
				sum_negative += abs(H(i, j));
				allPos = false;
			}
		}
	}

	float L = 255.0;
	float min = 0.0;
	float max = sum_positive * 255;

	float S = 1 / (2 * max(sum_negative, sum_positive));

	if (allPos) {

		// LOW PASS

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {

				dst(i, j) = L * (convo(i, j) - min) / (max - min);
			}
		}
	}
	else {	// HIGH FILTER

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {

				dst(i, j) = S * convo(i, j) + (floor)(L / 2);
			}
		}
	}

	return dst;

}

void centering_transform(Mat img) {
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat generic_frequency_domain_filter(Mat src, bool gauss, bool low_pass) {
	//convert input image to float image
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	//centering transformation
	centering_transform(srcf);

	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	//split into real and imaginary channels
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))

	//calculate magnitude and phase in floating point images mag and phi
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);

	//display the phase and magnitude images here
	for (int i = 0; i < srcf.rows; i++) {
		for (int j = 0; j < srcf.cols; j++) {
			mag.at<float>(i, j) = log(mag.at<float>(i, j) + 1);
		}
	}

	normalize(mag, mag, 0, 255, NORM_MINMAX, CV_8UC1);
	normalize(phi, phi, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("Magnitude", mag);
	imshow("Phase", phi);

	float aux;

	if (!gauss) {

		int R2 = 400;

		for (int i = 0; i < mag.rows; i++) {
			for (int j = 0; j < mag.cols; j++) {

				aux = pow((float)mag.rows / 2 - i, 2) + pow((float)mag.cols / 2 - j, 2);

				if (low_pass) {	// LOW-PASS

					if (aux > R2) {
						mag.at<float>(i, j) = 0;
					}
					else {
						mag.at<float>(i, j) = mag.at<float>(i, j);
					}
				}
				else {	// HIGH-PASS
					if (aux > R2) {
						mag.at<float>(i, j) = mag.at<float>(i, j);
					}
					else {
						mag.at<float>(i, j) = 0;
					}
				}
			}
		}
	}
	else {
		int A = 20;

		for (int i = 0; i < mag.rows; i++) {
			for (int j = 0; j < mag.cols; j++) {

				aux = -(pow(mag.rows / 2 - i, 2) + pow(mag.cols / 2 - j, 2)) / pow(A, 2);

				if (low_pass) {	// LOW-PASS
					mag.at<float>(i, j) = mag.at<float>(i, j) * exp(aux);

				}
				else {	// HIGH-PASS
					mag.at<float>(i, j) = mag.at<float>(i, j) * (1 - exp(aux));

				}
			}
		}
	}

	//store in real part in channels[0] and imaginary part in channels[1]
	//...
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			channels[0].at<float>(i, j) = mag.at<float>(i, j) * cos(phi.at<float>(i, j));
			channels[1].at<float>(i, j) = mag.at<float>(i, j) * sin(phi.at<float>(i, j));
		}
	}

	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	//inverse centering transformation
	centering_transform(dstf);

	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

	//Note: normalizing distorts the resut while enhancing the image display in the range [0,255].
	//For exact results (see Practical work 3) the normalization should be replaced with convertion:
	//dstf.convertTo(dst, CV_8UC1);

	return dst;
}

void fourier(Mat_<uchar> img, bool gauss, bool low_pass) {

	Mat dst = generic_frequency_domain_filter(img, gauss, low_pass);

	imshow("Original", img);

	if (gauss) {
		if (low_pass) {
			imshow("Gaussian - low-pass", dst);
		}
		else {
			imshow("Gaussian - high-pass", dst);
		}
	}
	else {
		if (low_pass) {
			imshow("Non-Gaussian - low-pass", dst);
		}
		else {
			imshow("Non-Gaussian - high-pass", dst);
		}
	}

	waitKey(0);
}

void medianFilter(Mat img, int w)
{
	imshow("Initial image", img);

	double t = (double)getTickCount();

	int center = w / 2;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			Mat kernel = Mat(w, w, CV_8UC1);
			kernel.at<uchar>(center, center) = img.at <uchar>(i, j);
			for (int ki=0; ki < w; ki++)
			{
				for (int kj = 0; kj < w; kj++)
				{
					auto ii = i - w / 2 + ki;
					auto ij = j - w / 2 + kj;
					if (isInside(img, i, j))
						kernel.at<uchar>(ki, kj) = img.at<uchar>(ii, ij);
					else
						kernel.at<uchar>(ki, kj) = 128;
					//kernel.at<uchar>(ki, kj) = (isInside(img, i, j)) ? img.at<uchar>(ii, ij) : 128;
				}
			}

			for (int k = 0; k < w*w; k++)
			{
				for (int l = 0; l < w * w; l++)
				{
					if (kernel.data[k] > kernel.data[l])
						swap(kernel.data[k], kernel.data[l]);
				}
			}
			img.at<uchar>(i, j) = kernel.data[w * w / 2];
		}
	}
	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("Median Time = %.3f [ms]\n", t * 1000);

	imshow("Median filter", img);
	waitKey(0);
}

void gaussianFilter2D(Mat_<uchar> img, int w) {

	int height = img.rows;
	int width = img.cols;

	double t = (double)getTickCount(); // Get the current time [ms]
	// … Actual processing …
	int k = w / 2;
	int x0 = w / 2;
	int y0 = w / 2;
	float sigma = (float)w / 6;

	Mat_<float> val(w, w);

	for (int u = 0; u < w; u++) {
		for (int v = 0; v < w; v++) {

			float dividend = (2.0f * PI * sigma * sigma);
			float first = (float)((u - x0) * (u - x0) + (v - y0) * (v - y0));
			float exponent = (float)first / (2 * sigma * sigma);

			val(u, v) = (float)(1 / dividend) * exp(-exponent);
		}
	}

	Mat_<float> convo = convolution(img, val);

	Mat res;
	convo.convertTo(res, CV_8UC1);

	// Get the current time again and compute the time difference [ms]
	t = ((double)getTickCount() - t) / getTickFrequency();
	// Display (in the console window) the processing time in [ms]
	printf("Gauss Time = %.3f [ms]\n", t * 1000);

	imshow("Initial image", img);
	imshow("2D Gaussian filter", res);
	waitKey(0);
}


int find_partition(float nr) {

	float pi = CV_PI;

	if ((nr >= pi / 8 && nr <= 3 * pi / 8) || (nr >= -7 * pi / 8 && nr <= -5 * pi / 8)) {
		return 1;
	}
	else if ((nr >= 3 * pi / 8 && nr <= 5 * pi / 8) || (nr >= -5 * pi / 8 && nr <= -3 * pi / 8)) {
		return 2;
	}
	else if ((nr >= 5 * pi / 8 && nr <= 7 * pi / 8) || (nr >= -3 * pi / 8 && nr <= -pi / 8)) {
		return 3;
	}
	else return 0;
}

void cannyEdgeDetection(Mat_<uchar> img) {

	int height = img.rows;
	int width = img.cols;

	// Gaussian filtering

	Mat_<float> gaussian_filter(3, 3);
	gaussian_filter = (1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0);
	Mat_<uchar> gauss = normalization(gaussian_filter, convolution(img, gaussian_filter));
	imshow("After Gaussian filter", gauss);



	Mat_<float> orientation(height, width);
	Mat_<uchar> dst1(height, width);
	Mat_<float> normalized_G(height, width);
	Mat_<uchar> dst2(height, width);


	int Sx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
	int Sy[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {

			int valX = 0;
			int valY = 0;

			for (int k = 0; k < 3; k++) {
				for (int p = 0; p < 3; p++) {

					valX += gauss(i - 3 / 2 + k, j - 3 / 2 + p) * Sx[k][p];
					valY += gauss(i - 3 / 2 + k, j - 3 / 2 + p) * Sy[k][p];
				}
			}

			float G = sqrt(valX * valX + valY * valY);
			G /= 4 * sqrt(2);

			dst1(i, j) = (int)G;
			dst2(i, j) = (int)G;


			normalized_G(i, j) = G;
			orientation(i, j) = atan2(valY, valX);
		}
	}


	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {

			int partition = find_partition(orientation.at<float>(i, j));

			float pixel = normalized_G.at<float>(i, j);

			int x1 = 0;
			int x2 = 0;
			int y1 = 0;
			int y2 = 0;

			switch (partition) {

			case 0:
				x1 = i;
				x2 = i;
				y1 = j - 1;
				y2 = j + 1;
				break;

			case 1:
				x1 = i + 1;
				x2 = i + 1;
				y1 = j - 1;
				y2 = j - 1;
				break;

			case 2:
				x1 = i + 1;
				x2 = i + 1;
				y1 = j;
				y2 = j;
				break;

			case 3:
				x1 = i - 1;
				x2 = i + 1;
				y1 = j - 1;
				y2 = j + 1;
				break;
			}

			float neigh1 = normalized_G.at<float>(x1, y1);
			float neigh2 = normalized_G.at<float>(x2, y2);

			if (!(pixel > neigh1 && pixel > neigh2)) {
				dst2(i, j) = 0;
			}
		}
	}

	imshow("Normalized gradient magnitude", dst1);
	imshow("Non-maxima suppression", dst2);
	waitKey(0);
}


int main()
{
	int op;
	do
	{
		{
			system("cls");
			destroyAllWindows();
			printf("Menu:\n");
			printf(" 1 - Open image\n");
			printf(" 2 - Open BMP images from folder\n");
			printf(" 3 - Image negative - diblook style\n");
			printf(" 4 - BGR->HSV\n");
			printf(" 5 - Resize image\n");
			printf(" 6 - Canny edge detection\n");
			printf(" 7 - Edges in a video sequence\n");
			printf(" 8 - Snap frame from live video\n");
			printf(" 9 - Mouse callback demo\n");
			printf(" 12 - lab1Ex2\n");
			printf(" 13 - lab1Ex3\n");
			printf(" 14 - lab1Ex4\n");
			printf(" 15 - lab1Ex5\n");
			printf(" 21 - lab2Ex1\n");
			printf(" 22 - lab2Ex2\n");
			printf(" 23 - lab2Ex3\n");
			printf(" 24 - lab2Ex4\n");
			printf(" 25 - lab2Ex5\n");
			printf(" 31 - lab3Ex1\n");
			printf(" 32 - lab3Ex2\n");
			printf(" 33 - lab3Ex3\n");
			printf(" 34 - lab3Ex4\n");
			printf(" 35 - lab3Ex5\n");
			printf(" 36 - lab3Ex6\n");
			printf(" 41 - lab4Ex1\n");
			printf(" 43 - lab4Ex3\n");
			printf(" 51 - lab5Ex1\n");
			printf(" 60 - lab6\n");
			printf(" 71 - lab7_Dilation\n");
			printf(" 711 - lab7_Dilation N Times\n");
			printf(" 72 - lab7_Erosion\n");
			printf(" 721 - lab7_Erosion N Times\n");
			printf(" 73 - lab7_Opening\n");
			printf(" 74 - lab7_Closing\n");
			printf(" 75 - lab7_Boundary\n");
			printf(" 76 - lab7_RegionFilling\n");
			printf(" 81 - lab8_Mean&StandardDeviation\n");
			printf(" 82 - lab8_BasicGlobalTresholding\n");
			printf(" 83 - lab8_BrightnessChange\n");
			printf(" 84 - lab8_ContrastChange\n");
			printf(" 85 - lab8_GammaCorrection\n");
			printf(" 86 - lab8_HistogramEqualization\n");
			printf(" 91 - lab9_MeanFilter\n");
			printf(" 92 - lab9_GaussianFilter\n");
			printf(" 93 - lab9_Laplace\n");
			printf(" 94 - lab9_HighPass\n");
			printf(" 101 - lab10_Gauss\n");
			printf(" 102 - lab10_Median\n");
			printf(" 111 - lab11_EdgeLinking\n");
			printf(" 0 - Exit\n\n");
			printf("Option: ");
			scanf("%d", &op);
		}

		int value;
		int histogram[256];
		float pdf[256];
		Mat_<uchar> cameraManImg = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
		Mat_<uchar> saturnImg = imread("Images/saturn.bmp", IMREAD_GRAYSCALE);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 12:
				lab1Ex2();
				break;
			case 13:
				printf("Enter value: ");
				scanf("%d", &value);
				lab1Ex3(value);
				break;
			case 14:
				printf("Enter value: ");
				scanf("%d", &value);
				lab1Ex4(value);
				break;
			case 15:
				lab1Ex5();
				break;
			case 21:
				lab2Ex1();
				break;
			case 22:
				lab2Ex2();
				break;
			case 23:
				lab2Ex3();
				break;
			case 24:
				lab2Ex4();
				break;
			case 25:
			{
				Mat_<uchar> img = imread("Images/kids.bmp", 0);
				int i, j;
				printf("\nEnter i coordinate of the pixel: ");
				scanf("%d", &i);
				printf("\nEnter j coordinate of the pixel: ");
				scanf("%d", &j);
				if(isInside(img, i, j))
					printf("The pixel is in the image");
				else
					printf("The pixel is not in the image");
				Sleep(2000);
				break;
			}
			case 31:
				lab3Ex1(cameraManImg, histogram);
				break;
			case 32:
				lab3Ex2(cameraManImg, pdf, histogram);
				break;
			case 33:
				lab3Ex3(histogram, 256, 256);
				break;
			case 34:
				lab3Ex4(cameraManImg, histogram, 64);
				break;
			case 35:
				//multiLevelTh(cameraManImg, histogram);
				lab3Ex5(cameraManImg, 5, 0.0003f);
				//Sleep(3000);
				break;
			case 36:
				
				lab3Ex6(saturnImg);
				break;
			case 41:
				lab4Ex1_BFS();
				break;
			case 43:
				lab4Ex3();
				break;
			case 51:
				lab5Ex1();
				break;
			case 60:
				lab6_borderTracing();
				break;
			case 71: {
				Mat_<uchar> img = imread("Images/Morphological_Op_Images/3_Open/cel4thr3_bw.bmp", IMREAD_GRAYSCALE);
				Mat_<uchar> struct_elem = generate_StructElem(5);
				Mat_<uchar> dst = dilation(img, struct_elem);
				imshow("Initial image", img);
				imshow("Dilation", dst);
				waitKey(0);
				break;
			}
			case 711: {
				Mat_<uchar> img = imread("Images/Morphological_Op_Images/3_Open/cel4thr3_bw.bmp", IMREAD_GRAYSCALE);
				Mat_<uchar> struct_elem = generate_StructElem(5);
				int n;
				printf("\nDati n: ");
				scanf("%d", &n);
				Mat_<uchar> dst = dilation(img, struct_elem);
				for (int i = 0; i < n; i++)
				{
					dst = dilation(dst, struct_elem);
				}
				imshow("Initial image", img);
				imshow("Dilation N Times", dst);
				waitKey(0);
				break;
			}
			case 72:
			{
				Mat_<uchar> img = imread("Images/Morphological_Op_Images/3_Open/cel4thr3_bw.bmp", IMREAD_GRAYSCALE);
				Mat_<uchar> struct_elem = generate_StructElem(5);
				Mat_<uchar> dst = erosion(img, struct_elem);
				imshow("Initial image", img);
				imshow("Erosion", dst);
				waitKey(0);
				break;
			}
			case 721:
			{
				Mat_<uchar> img = imread("Images/Morphological_Op_Images/3_Open/cel4thr3_bw.bmp", IMREAD_GRAYSCALE);
				Mat_<uchar> struct_elem = generate_StructElem(5);
				int n;
				printf("\nDati n: ");
				scanf("%d", &n);
				Mat_<uchar> dst = erosion(img, struct_elem);
				for (int i = 0; i < n; i++)
				{
					dst = erosion(dst, struct_elem);
				}
				imshow("Initial image", img);
				imshow("Erosion N Times", dst);
				waitKey(0);
				break;
			}
			case 73:
			{
				Mat_<uchar> img = imread("Images/Morphological_Op_Images/3_Open/cel4thr3_bw.bmp", IMREAD_GRAYSCALE);
				Mat_<uchar> struct_elem = generate_StructElem(5);
				opening(img, struct_elem);
				break;
			}
			case 74:
			{
				Mat_<uchar> img = imread("Images/Morphological_Op_Images/4_Close/phn1thr1_bw.bmp", IMREAD_GRAYSCALE);
				Mat_<uchar> struct_elem = generate_StructElem(5);
				closing(img, struct_elem);
				break;
			}
			case 75:
			{
				Mat_<uchar> img = imread("Images/Morphological_Op_Images/5_BoundaryExtraction/wdg2thr3_bw.bmp", IMREAD_GRAYSCALE);
				Mat_<uchar> struct_elem = generate_StructElem(5);
				Mat_<uchar> dst = boundaryExtraction(img, struct_elem);
				imshow("Initial image", img);
				imshow("Boundary", dst);
				waitKey(0);
				break;
			}
			case 76:
			{
				Mat_<uchar> img = imread("Images/Morphological_Op_Images/6_RegionFilling/wdg2ded1_bw.bmp", IMREAD_GRAYSCALE);
				Mat_<uchar> struct_elem = generate_StructElem(5);
				Mat_<uchar> dst = regionFilling(img, struct_elem);
				imshow("Initial image", img);
				imshow("Region Filling", dst);
				waitKey(0);
				break;
			}
			case 81:
			{
				Mat_<uchar> img = imread("Images/lab8/wheel.bmp", IMREAD_GRAYSCALE);
				int hist[256];
				computeHistogram1(img, hist, 256);
				int meanVl = meanValue(img, hist);
				int stdDeviation = standardDeviation(img, meanVl, hist);
				printf("Mean value: %d\nStandard deviation: %d\n", meanValue, stdDeviation);
				imshow("Initial Image",img);
				waitKey(0);
				break;
			}
			case 82:
			{
				Mat_<uchar> img = imread("Images/lab8/wheel.bmp", IMREAD_GRAYSCALE);
				int hist[256];
				computeHistogram1(img, hist, 256);
				lab8_globalTresholding(img, hist);
				break;
			}
			case 85:
			{
				Mat_<uchar> img = imread("Images/lab8/wilderness.bmp", IMREAD_GRAYSCALE);
				//int hist[256];
				//computeHistogram1(img, hist, 256);
				gamma_correction(img, 0.2, 3);
				break;
			}
			case 84:
			{
				Mat_<uchar> img = imread("Images/lab8/wheel.bmp", IMREAD_GRAYSCALE);
				int hist[256];
				computeHistogram1(img, hist, 256);
				int g_outMin, g_outMax;
				printf("Give min max\n");
				scanf("%d %d", &g_outMin, &g_outMax);
				contrastChange(img, g_outMin, g_outMax, hist);
				imshow("Initial Image", img);
				waitKey(0);
				break;
			}
			case 91:
			{
				Mat_<uchar> img = imread("Images/lab9/cameraman.bmp", IMREAD_GRAYSCALE);
				Mat_<float> mean_filter(3, 3);
				mean_filter.setTo(1.0 / 9.0);
				Mat_<uchar> dst = normalization(mean_filter, convolution(img, mean_filter));
				imshow("Mean filter", dst);
				imshow("Initial", img);
				waitKey(0);
				break;
			}
			case 92:
			{
				Mat_<uchar> img = imread("Images/lab9/cameraman.bmp", IMREAD_GRAYSCALE);
				Mat_<float> gaussian_filter(3, 3);
				gaussian_filter = (1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0);
				Mat_<uchar> dst = normalization(gaussian_filter, convolution(img, gaussian_filter));
				
				imshow("Gaussian filter", dst);
				waitKey(0);
				break;
			}
			case 93:
			{
				Mat_<uchar> img = imread("Images/lab9/cameraman.bmp", IMREAD_GRAYSCALE);
				Mat_<float> laplace_filter = (Mat_<float>(3, 3) << 0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0);
				Mat_<uchar> dst = normalization(laplace_filter, convolution(img, laplace_filter));
				
				imshow("Laplace filter", dst);
				waitKey(0);
				break;
			}
			case 94:
			{
				Mat_<uchar> img = imread("Images/lab9/cameraman.bmp", IMREAD_GRAYSCALE);
				Mat_<float> high_pass_filter = (Mat_<float>(3, 3) << 0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0);
				Mat_<uchar> dst = normalization(high_pass_filter, convolution(img, high_pass_filter));

				imshow("High pass filter", dst);
				waitKey(0);
				break;
			}
			case 101:
			{
				Mat_<uchar> img = imread("Images/Noise_Images/portrait_Gauss1.bmp", IMREAD_GRAYSCALE);
				gaussianFilter2D(img, 5);
				break;
			}
			case 102:
			{
				Mat_<uchar> img = imread("Images/Noise_Images/portrait_Salt&Pepper1.bmp", IMREAD_GRAYSCALE);
				medianFilter(img, 5);
				//waitKey(0);
				break;
			}
			case 111:
			{
				Mat_<uchar> img = imread("Images/saturn.bmp", IMREAD_GRAYSCALE);
				cannyEdgeDetection(img);
				//waitKey(0);
				break;
			}
			
			case 9:
				testMouseClick();
				break;
		}
	}
	while (op!=0);
	return 0;
}