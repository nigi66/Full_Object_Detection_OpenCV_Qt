#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QTabWidget>
#include <QPushButton>
#include <QComboBox>
#include <QTimer>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

using namespace std;
using namespace cv;


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    QTabWidget *tabs;

    // Image processing
    QWidget *imageTab;
    QLabel *imageDisplay;
    QPushButton *loadImageButton, *detectImageButton;
    QComboBox *imageMethodsDropdown;

    cv::Mat originalImage;       // Store the original image
    cv::Mat currentImage;        // Store the current image after filtering

    void loadImage();
    void detectInImage();
    void displayImage(const cv::Mat &image);

    void detectHaarCascades(cv::Mat &image);
    void detectYOLO(cv::Mat &frame);
    void detectSSD(Mat &image);
    void detectHOG(Mat &image);


    // Video Processing;
    QWidget *videoTab;
    QLabel *videoDisplay;
    QPushButton *loadVideoButton, *detectVideoButton;
    QComboBox *videoMethodsDropdown;

    void loadVideo();
    void detectInVideo();
    void processVideoFrame();


};
#endif // MAINWINDOW_H
