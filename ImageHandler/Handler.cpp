/*
Date:
	10/19/2018	directory traverse, normalization, trim structure, make robust
	10/18/2018	image read, write, add watermark
Author: Zixuan Zhang
Function: read all image from a folder (no sub folder), resize to specified size (keep ratio, add padding), add random watermark, save to specified folder
*/

#define WATERMARK_TEXT "Desung"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>
//#include <queue>
#include <cstdlib>
#include <ctime>
using namespace cv;
using namespace std;
namespace fs = std::filesystem;

float alpha = 0.5f;
int imgSize = 750;
bool needWatermark = true;
bool replaceOldFile = false;
bool randomNaming = true;

/*
Function:
	Rotate an image (refer: http://opencv-code.com/quick-tips/how-to-rotate-image-in-opencv/)
*/
void rotate(Mat src, double angle, Mat dst) {
	Point2f pt(src.cols / 2.f, src.rows / 2.f);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);

	warpAffine(src, dst, r, Size(src.cols, src.rows));	// len

	// get rotation matrix for rotating the image around its center in pixel coordinates
	/*cv::Point2f center((src.cols - 1) / 2.0, (src.rows - 1) / 2.0);
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	// determine bounding rectangle, center not relevant
	cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
	// adjust transformation matrix
	rot.at<double>(0, 2) += bbox.width / 2.0 - src.cols / 2.0;
	rot.at<double>(1, 2) += bbox.height / 2.0 - src.rows / 2.0;

	cv::Mat dst;
	cv::warpAffine(src, dst, rot, bbox.size());*/
}

/*
Function:
	generate random color (source: http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/random_generator_and_text/random_generator_and_text.html#drawing-2)
Input:
	rng: RNG, random number generator with seed
*/
static Scalar randomColor(RNG rng) {
	int icolor = (unsigned)rng;
	return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

/*
Function:
	Calculate mask of a RBG/3-channel imgae (refer: https://docs.opencv.org/3.4.3/db/da5/tutorial_how_to_scan_images.html)
Input:
	img: Mat, the image need to be calculate mask
*/
void calculateMask(Mat img, Mat mask) {
	// accept only char type matrices
	CV_Assert(mask.depth() == CV_8U);
	const int channels = mask.channels();
	MatIterator_<Vec3b> it, end;
	for (it = mask.begin<Vec3b>(), end = mask.end<Vec3b>(); it != end; ++it) {
		if (!(*it)[0] && !(*it)[1] && !(*it)[2]) {	// All zeros (black background) -> 1 (keep background)
			(*it)[0] = 1;
			(*it)[1] = 1;
			(*it)[2] = 1;
		} else {	// text -> black (0)
			(*it)[0] = 0;
			(*it)[1] = 0;
			(*it)[2] = 0;
		}
	}
	/*namedWindow("reversed mask image", CV_WINDOW_AUTOSIZE);
	imshow("reversed mask image", mask);
	waitKey(0);*/
}

/*
Function:
	add watermark to the picture with random position, font, font size, thickness, and angle (refer: https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/)
Input:
	backgroundImg: Mar, the image need watermark
	rng: RNG, random number generator with seed (not fixed, otherwise no random)
	alpha: double, alpha value (transparency) of the watermark
	cleanBeforeAdd: bool, indicate whether watermark area needs to be erased in background image before blending
*/
void addWatermark(Mat backgroundImg, RNG rng, double alpha, bool cleanBeforeAdd) {
	// Initialization
	int rowCount = backgroundImg.rows;
	int colCount = backgroundImg.cols;
	Mat textImg = Mat::zeros(rowCount, colCount, backgroundImg.type());

	// Generate the random values
	int fontType = rng.uniform(0, 8);
	double fontScale = rng.uniform(0, 100)*0.05 + 0.1;
	int thickness = rng.uniform(1, 5);
	Scalar color = randomColor(rng);
	double angle = rng.uniform(-90., 90.);
	int baseline = 0;

	Size textSize = getTextSize(WATERMARK_TEXT, fontType, 2, thickness, &baseline);	// width, height: double, size not considering thickness
	//double x = rng.uniform(textSize.width / 2. + thickness, (colCount - textSize.width) / 2. - thickness);	// position to add watermark
	//double y = rng.uniform(textSize.height / 2. + thickness, (rowCount - textSize.height) / 2. - thickness);	// ensure no cutting

	// coordinates of Bottom-left corner of the text string in the image.
	double x = rng.uniform(thickness, colCount - 2 * thickness - textSize.width);	// position to add watermark
	double y = rng.uniform(2 * thickness + textSize.height, rowCount - thickness);	// ensure no cutting

	//cout << x << " " << y << " " << fontType << " " << fontScale << " " << thickness << " " << color << " " << angle << endl;
	// Put text to text image
	putText(textImg, WATERMARK_TEXT, Point2d(x, y), fontType, 2, randomColor(rng), thickness);
	/*namedWindow("text image", CV_WINDOW_AUTOSIZE);
	imshow("text image", textImg);
	waitKey(0);*/

	// Rotate text image
	rotate(textImg, angle, textImg);
	/*namedWindow("rotate text image", CV_WINDOW_AUTOSIZE);
	imshow("rotate text image", textImg);
	waitKey(0);*/


	// Add watermark to original image
	if (cleanBeforeAdd) {
		Mat mask = textImg.clone();
		calculateMask(textImg, mask);// = Mat::zeros(rowCount, colCount, textImg.type());
		/*cvtColor(textImg, mask, CV_RGB2GRAY);
		threshold(mask, mask, 0, 255, THRESH_BINARY);
		mask = mask < 100;*/
		//Scalar::all(1.0) - mask
		multiply(mask, backgroundImg, backgroundImg);	// make background of text white
		/*namedWindow("back image", CV_WINDOW_AUTOSIZE);
		imshow("back image", backgroundImg);
		waitKey(0);*/
	}
	//add(alpha * textImg, backgroundImg, backgroundImg);
	backgroundImg += alpha * textImg;
}

/*
Function:
	normalize size of an image to the size of another while keep the ratio (padding: black)
Input:
	src: Mat, the source image
	dst: Mat, the destination (standard) image
*/
void normalizeSize(Mat src, Mat dst) {
	int oriWidth = src.cols;
	int oriHeight = src.rows;
	int targetWidth = dst.cols;
	int targetHeight = dst.rows;

	if (oriWidth == targetWidth && oriHeight == targetHeight) {
		src.copyTo(dst);
	} else {
		Rect roi;	// original image area in standard image

		float widthRatio = float(targetWidth) / float(oriWidth);	// can enlarge / shrink to xx of original
		float heightRatio = float(targetHeight) / float(oriHeight);	// can enlarge / shrink to xx of original
		if (widthRatio > heightRatio) {	// choose the min ratio to ensure the min (both) fit
			// Satisfy height first: height fill, width change according to height ratio (have residual)
			roi.height = targetHeight;
			roi.width = int(floor(oriWidth * heightRatio));
			roi.x = (targetWidth - roi.width) / 2;
			roi.y = 0;
		} else {
			// Satisfy width first: width fill, height change according to width radio (have residual)
			roi.height = int(floor(oriHeight * widthRatio));
			roi.width = targetWidth;
			roi.x = 0;
			roi.y = (targetHeight - roi.height) / 2;
		}

		resize(src, dst(roi), roi.size(), 0, 0, INTER_LANCZOS4);
	}
}

/*
Function:
	get directory path of a file/directory
Input: 
	path: string, path of a file/directory
Output:
	dirPath: string, path of the parent of the file/directory
!!only works in Windows!!
*/
string getDir(string path) {
	size_t pos = path.find_last_of('\\');
	string dirPath = pos == -1 ? path : path.substr(0, pos);	// if root: root

	return dirPath;
}

/*
Function: 
	get file name of a full path
Input:
	filePath: string, path of a file
Output:
	fileName: string, name of the file
!!only works in Windows!! no error handling, assume file path
*/
string getFileName(string filePath) {
	size_t pos = filePath.find_last_of('\\');	// if -1: not found, may be relative path
	string fileName = (pos == -1 ? filePath : filePath.substr(pos + 1));

	return fileName;
}

static const char alphanum[] =
"0123456789"
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz";

int stringLength = sizeof(alphanum) - 1;

unsigned int seed;

/*
Function:
	generate random string, using characters in global char array: alphanum, global variable seed to determine whether the seed should be reset (refer: http://www.cplusplus.com/forum/windows/88843/)
Input:
	length: int, length of generated string
Output:
	resStr: string, generated random string
*/
string getRandomStr(int length) {
	// Get seed: if the same as last time: do not set (otherwise same string generated)
	if (time(0) != seed) {
		seed = time(0);
		srand(seed);
	}

	string resStr;
	for (int i = 0; i < length; i++) {
		resStr += alphanum[rand() % stringLength];
	}

	return resStr;
}

/*
Function:
	add random string to the end of filename, keeping the extension
Input:
	fileName: string, name of the file
Output:
	newFileName: string, name of revised file
*/
string reviseFileName(string fileName) {
	int pos = fileName.rfind('.');
	string newFileName = fileName.substr(0, pos) + '-' + getRandomStr(3) + fileName.substr(pos, fileName.length());

	return newFileName;
}

/*
Function:
	handle images in the directory, and sub directories without keeping the directory structure
Input:
	inPath: string, path of input directory or file
	outPath: string, path of output directory
*/
void handlePath(string rootInPath, string inPath, string outPath) {
	//fs::path out = outPath;

	// Traverse the directory
	for (auto & p : fs::directory_iterator(inPath)) {
		// Recursively traverse the sub-directory
		if (fs::is_directory(p)) {
			cout << "Handling directory: " << p.path().string() << endl;
			handlePath(rootInPath, p.path().string(), outPath);
		}
		// Only handle regular file
		if (!fs::is_regular_file(p)) {
			cout << "Not regular file: " << p.path().string() << endl;
			continue;
		}
		string inFilePath = p.path().string();
		// hard to keep directory structure
		//fs::path inFileName = p.path().filename();
		//fs::path outFilePath = out / inFileName;

		string outFilePath = inFilePath;
		outFilePath.replace(outFilePath.find(rootInPath), rootInPath.length(), outPath);
		fs::path out = outFilePath;
		if (randomNaming) {
			string currentNaming = out.filename().string();
			out.replace_filename(reviseFileName(currentNaming));
			outFilePath = out.string();
		}
		fs::create_directories(out.parent_path());

		cout << inFilePath << " " << outFilePath << endl;

		// Handle existing file
		if (!replaceOldFile && fs::exists(outFilePath)) {	// outFilePath
			cout << "Already exists: " << outFilePath << endl;		// outFilePath.string()
			continue;
		}

		cout << "Handling: " << inFilePath << endl;
		
		// Get input image
		Mat img = imread(inFilePath, CV_LOAD_IMAGE_COLOR);
		if (!img.data) {	// only handle image
			cout << inFilePath << ": NO image data." << endl;
			continue;
		}

		// Handle the image
		Mat standardImg = Mat::zeros(imgSize, imgSize, img.type());

		// Normalize the size
		normalizeSize(img, standardImg);

		// Add watermark (random, skewed)
		if (needWatermark) {
			addWatermark(standardImg, cvRNG(getTickCount()), alpha, false);
		}

		/*namedWindow("standard image", CV_WINDOW_AUTOSIZE);
		imshow("standard image", standardImg);
		waitKey(0);*/

		// Save to ouput image
		imwrite(outFilePath, standardImg);	// outFilePath.string()
		cout << "Finished: " << outFilePath << endl;	// outFilePath
	}
}

int main(int argc, char** argv) {
	// TODO: how to run continously: save previous status (quick) / detect duplicate (?? litter dependency, slow)

	// Get input path and output path
	string inPath;
	string outPath;
	if (argc == 2) {	// pay attention to relative path
		inPath = argv[1];
		outPath = getDir(inPath) + "\\handledImage";
	} else if (argc == 3) {
		inPath = argv[1];
		outPath = argv[2];
		/*if (!fs::is_directory(outPath)) {
			printf("%s not exist or is not directory!\n ", argv[2]);
			return -1;
		}*/
	} else {
		printf("useage: %s <input image directory> [<output image directory>]\n ", argv[0]);
		return -1;
	}
	//cout << inPath << endl << outPath << endl;
	//string inPath = "C:\\Users\\User\\source\\repos\\ImageHandler\\ImageHandler\\image";	// must in Windows, 
	//string outPath = getDir(inPath) + "\\handledImage";
	if (!fs::is_directory(inPath)) {
		printf("%s not exist or is not directory!\n ", argv[1]);
		return -1;
	}
	// create the output directory if not exist
	fs::create_directories(outPath);

	// Set random number generator, parameters
	RNG rng(0xFFFFFFFF);

	if (randomNaming) {
		replaceOldFile = false;
	}

	handlePath(inPath, inPath, outPath);

	// file url after upload: https://cdn.shopify.com/s/files/1/0025/7962/8143/files/image_name.jpg?5125681438951551083

	return 0;
}