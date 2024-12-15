#include "mainwindow.h"
#include "ui_mainwindow.h"


#include <QMessageBox>
#include <QFileDialog>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    tabs = new QTabWidget(this);

    // Image processing Tab
    imageTab = new QWidget();
    imageDisplay = new QLabel();
    loadImageButton = new QPushButton("Load Image");
    detectImageButton = new QPushButton("Detect");
    imageMethodsDropdown = new QComboBox();
    imageMethodsDropdown->addItems({"Haar Cascades", "YOLO", "SSD", "HOG+SVM"});
    QVBoxLayout *imageLayout = new QVBoxLayout();
    imageLayout->addWidget(loadImageButton);
    imageLayout->addWidget(imageMethodsDropdown);
    imageLayout->addWidget(detectImageButton);
    imageLayout->addWidget(imageDisplay);
    imageTab->setLayout(imageLayout);
    tabs->addTab(imageTab, "Image Processing");



    // Video Processing Tab
    videoTab = new QWidget();
    videoDisplay = new QLabel();
    loadVideoButton = new QPushButton("Load Video");
    detectVideoButton = new QPushButton("Detect");
    videoMethodsDropdown = new QComboBox();
    videoMethodsDropdown->addItems({"Haar Cascades", "YOLO", "SSD", "HOG+SVM"});
    QVBoxLayout *videoLayout = new QVBoxLayout();
    videoLayout->addWidget(loadVideoButton);
    videoLayout->addWidget(videoMethodsDropdown);
    videoLayout->addWidget(detectVideoButton);
    videoLayout->addWidget(videoDisplay);
    videoTab->setLayout(videoLayout);
    tabs->addTab(videoTab, "Video Processing");

    setCentralWidget(tabs);

    // Connect signals to slots
    connect(loadImageButton, &QPushButton::clicked, this, &MainWindow::loadImage);
    connect(detectImageButton, &QPushButton::clicked, this, &MainWindow::detectInImage);
    connect(loadVideoButton, &QPushButton::clicked, this, &MainWindow::loadVideo);
    connect(detectVideoButton, &QPushButton::clicked, this, &MainWindow::detectInVideo);


}


MainWindow::~MainWindow()
{
    delete ui;
}


// Image Processing Slot
void MainWindow::loadImage() {
    QString filePath = QFileDialog::getOpenFileName(this, "Open Image File", "", "Images(*.png *.jpg *.jpeg *.bmp *.tiff)");

    if (filePath.isEmpty()){
        return;
    }

    originalImage = imread(filePath.toStdString(), IMREAD_COLOR);
    if (originalImage.empty()){
        return;
    }
    currentImage = originalImage.clone();
    displayImage(currentImage);
}

void MainWindow::displayImage(const cv::Mat &image){
    cv::Mat rgbImage;
    cvtColor(image, rgbImage, COLOR_BGR2RGB);

    QImage qImage(rgbImage.data, rgbImage.cols, rgbImage.rows, rgbImage.step, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(qImage).scaled(imageDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);

    imageDisplay->setPixmap(pixmap);
}



void MainWindow::detectInImage() {

    Mat result = originalImage.clone();

    QString selectedMethod = imageMethodsDropdown->currentText();

    if (selectedMethod == "Haar Cascades"){
        detectHaarCascades(result);
    }else if (selectedMethod == "YOLO"){
        detectYOLO(result);
    }else if (selectedMethod == "SSD"){
        detectSSD(result);
    }else if (selectedMethod == "HOG+SVM"){
        detectHOG(result);
    }

    displayImage(result);
}


VideoCapture videoCapture;
QTimer *videoTimer;

// Video Processing Slot
void MainWindow::loadVideo() {
    QString filePath = QFileDialog::getOpenFileName(this, "Open Video", "", "Videos (*.mp4 *.avi *.mkv)");
    if (filePath.isEmpty()) return;

    videoCapture.open(filePath.toStdString());
    if (!videoCapture.isOpened()){
        QMessageBox::warning(this, "Error", "Failed to load video.");
        return;
    }

    if (!videoTimer){
        videoTimer = new QTimer(this);
        connect(videoTimer, &QTimer::timeout, this, &MainWindow::processVideoFrame);
    }
    videoTimer->start(30);
}

void MainWindow::processVideoFrame() {
    cv::Mat frame;
    if (!videoCapture.read(frame)) {
        videoTimer->stop();  // Stop the timer if no more frames
        return;
    }

    cv::Mat rgbFrame;
    cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
    QImage qImage((uchar*)rgbFrame.data, rgbFrame.cols, rgbFrame.rows, rgbFrame.step, QImage::Format_RGB888);

    videoDisplay->setPixmap(QPixmap::fromImage(qImage).scaled(videoDisplay->size(), Qt::KeepAspectRatio));
}


void MainWindow::detectInVideo() {

    if (!videoCapture.isOpened()){
        QMessageBox::warning(this, "Error", "Unable to load the video.");
        return;
    }
    Mat frame;
    while (videoCapture.read(frame)){
        QString selectedMethod = videoMethodsDropdown->currentText();

        if (selectedMethod == "Haar Cascades"){
            detectHaarCascades(frame);
        }else if (selectedMethod == "YOLO"){
            detectYOLO(frame);
        }else if (selectedMethod == "SSD"){
            detectSSD(frame);
        }else if (selectedMethod == "HOG+SVM"){
            detectHOG(frame);
        }

        Mat rgbFrame;
        cvtColor(frame, rgbFrame, COLOR_BGR2RGB);
        QImage qImage((uchar*)rgbFrame.data, rgbFrame.cols, rgbFrame.rows, rgbFrame.step, QImage::Format_RGB888);
        videoDisplay->setPixmap(QPixmap::fromImage(qImage).scaled(videoDisplay->size(), Qt::KeepAspectRatio));

    }

}



void MainWindow::detectHaarCascades(cv::Mat &image){

    CascadeClassifier faceCascade;
    if (!faceCascade.load("C:\\opencv-4.5.4\\build\\install\\etc\\haarcascades\\haarcascade_frontalface_default.xml")){
        QMessageBox::warning(this, "Error", "Failed to load Haar Cascade file");
        return;
    }

    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    vector<Rect> faces;
    faceCascade.detectMultiScale(grayImage, faces);

    for (const auto &face : faces){
        rectangle(image, face, Scalar(0, 255, 0), 2);
    }
}



void MainWindow::detectYOLO(cv::Mat &frame) {
    // Paths to YOLO files
    std::string modelConfiguration = "C:\\Users\\Me\\Documents\\opencv_cpp\\full_detection\\full_detection\\yolov4.cfg";
    std::string modelWeights = "C:\\Users\\Me\\Documents\\opencv_cpp\\full_detection\\full_detection\\yolov4.weights";
    std::string classNamesFile = "C:\\Users\\Me\\Documents\\opencv_cpp\\full_detection\\full_detection\\coco.names";

    static dnn::Net net = dnn::readNet(modelWeights, modelConfiguration);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    static vector<string> classNames;

    if(classNames.empty()){
        ifstream classFile(classNamesFile);
        if (!classFile.is_open()){
            QMessageBox::warning(this, "Error", "Failed to load class names file");
            return;
        }
        string line;
        while (getline(classFile, line)){
            classNames.push_back(line);
        }
    }

    Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, Size(256, 256), Scalar(0, 0, 0), true, false);

    net.setInput(blob);

    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float confidenceThreshold = 0.5f;
    float nmsTHreshold = 0.4f;

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for(const auto &output : outputs){
        for (int i=0; i<output.rows; ++i){
            const float *data = output.ptr<float>(i);
            float confidence = data[4];

            if (confidence > confidenceThreshold){
                Mat scores = Mat(1, classNames.size(), CV_32FC1, (void*)(data + 5));
                Point classIdPoint;
                double maxClassScores;
                minMaxLoc(scores, 0, &maxClassScores, 0, &classIdPoint);

                if (maxClassScores > confidenceThreshold){
                    int centerX = static_cast<int>(data[0] * frame.cols);
                    int centerY = static_cast<int>(data[1] * frame.rows);
                    int width = static_cast<int>(data[2] * frame.cols);
                    int height = static_cast<int>(data[3] * frame.cols);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back(static_cast<float>(maxClassScores));
                    boxes.push_back(Rect(left, top, width, height));
                }
            }

        }
    }

    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsTHreshold, indices);

    for (int idx: indices){
        const Rect &box = boxes[idx];
        int classId = classIds[idx];
        string label = classNames[classId] + ": " + format("%.2f", confidences[idx]);
        rectangle(frame, box, Scalar(0, 255, 0), 2);
        putText(frame, label, Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }
}



void MainWindow::detectSSD(Mat &image){

    static dnn::Net net = dnn::readNetFromCaffe(
                "C:\\Users\\Me\\Documents\\opencv_cpp\\full_detection\\full_detection\\deploy.prototxt",
                "C:\\Users\\Me\\Documents\\opencv_cpp\\full_detection\\full_detection\\res10_300x300_ssd_iter_140000.caffemodel");

    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    Mat blob = dnn::blobFromImage(image, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);

    net.setInput(blob);

    Mat detections = net.forward();

    float *data = (float *)detections.data;
    for (size_t i = 0; i < detections.total(); i += 7) {
        float confidence = data[i + 2];
        if (confidence > 0.5) {
            int x1 = static_cast<int>(data[i + 3] * image.cols);
            int y1 = static_cast<int>(data[i + 4] * image.rows);
            int x2 = static_cast<int>(data[i + 5] * image.cols);
            int y2 = static_cast<int>(data[i + 6] * image.rows);

            // Draw a rectangle around detected objects
            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

            // Add a label with confidence
            std::string label = cv::format("Confidence: %.2f", confidence);
            cv::putText(image, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }
}



void MainWindow::detectHOG(Mat &image){

    static cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    // Detect pedestrians
    std::vector<cv::Rect> detections;
    std::vector<double> weights;
    hog.detectMultiScale(image, detections, weights, 0, cv::Size(8, 8), cv::Size(16, 16), 1.05, 2);

    // Draw rectangles for detected pedestrians
    for (size_t i = 0; i < detections.size(); ++i) {
        cv::Rect rect = detections[i];
        double confidence = weights[i];
        if (confidence > 0.5) {
            cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);

            std::string label = cv::format("Confidence: %.2f", confidence);
            cv::putText(image, label, cv::Point(rect.x, rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }

}

