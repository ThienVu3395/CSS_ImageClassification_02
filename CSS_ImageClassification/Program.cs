//using CSS_ImageClassification;
//using Microsoft.ML;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using Keras;
using Keras.Models;
using Keras.Layers;
using Keras.Optimizers;
using Keras.Utils;
using Numpy;
using Tensorflow;
using Keras.Datasets;
using System;
using Python.Runtime;
using System.IO;
using System.Drawing.Text;
using Google.Protobuf.WellKnownTypes;
using System.Threading;

//// ====================================== Training Model
int batch_size = 128;
int num_classes = 41;
int epochs = 50;
int img_size = 28;
Shape input_shape = (img_size, img_size, 1);
Size size = new Size(img_size, img_size);
string folderTrainPath = @"D:\0.Projects\CSS_ImageClassification-master\CSS_ImageClassification\Train\";
string folderTestPath = @"D:\0.Projects\CSS_ImageClassification-master\CSS_ImageClassification\Test\";
string modelJsonPath = @"D:\0.Projects\CSS_ImageClassification-master\CSS_ImageClassification\Model\myClassificationModel.json";
string modelWeightPath = @"D:\0.Projects\CSS_ImageClassification-master\CSS_ImageClassification\Model\myClassificationModel.h5";

//List<NDarray> train_x_pre = new List<NDarray>();
//List<NDarray> train_y_pre = new List<NDarray>();
//Console.WriteLine("========================== Load Training Data =================================");
//foreach (string folder in Directory.EnumerateDirectories(folderTrainPath, "*"))
//{
//    Console.WriteLine($">>>>>>>>>>>>>>>>>>>>> Load Folder : {Path.GetFileName(folder)}");
//    int label = int.Parse(Path.GetFileName(folder));
//    NDarray label_y = (NDarray)label;
//    foreach (string file in Directory.EnumerateFiles(folder, "*.jpg"))
//    {
//        Mat img = Cv2.ImRead(file, ImreadModes.Color);
//        Mat imgGray = new Mat();
//        Cv2.CvtColor(img, imgGray, ColorConversionCodes.BGR2GRAY);
//        Mat imgThres = new Mat();
//        Cv2.Threshold(imgGray, imgThres, 150, 250, ThresholdTypes.Otsu);
//        Mat imgResize = new Mat();
//        Cv2.Resize(imgThres, imgResize, size , interpolation : InterpolationFlags.Area);
//        imgResize.GetArray(out byte[] plainArray);
//        NDarray imgNDArray = np.array(plainArray, dtype: np.uint8).reshape(img_size,img_size);
//        train_x_pre.add(imgNDArray);
//        train_y_pre.add(label_y);
//    }
//}

//NDarray train_x = np.array(train_x_pre);
//NDarray train_y = np.array(train_y_pre);
//train_x = train_x.reshape(train_x.shape[0], img_size, img_size, 1);
//train_x = train_x.astype(np.float32);
//train_y = Util.ToCategorical(train_y, num_classes);

//List<NDarray> test_x_pre = new List<NDarray>();
//List<NDarray> test_y_pre = new List<NDarray>();
//Console.WriteLine("========================== Load Testing Data =================================");
//foreach (string folder in Directory.EnumerateDirectories(Path.Combine(folderTestPath, "Clean_02"), "*"))
//{
//    Console.WriteLine($">>>>>>>>>>>>>>>>>>>>> Load Folder : {Path.GetFileName(folder)}");
//    int label = int.Parse(Path.GetFileName(folder));
//    NDarray label_y = (NDarray)label;
//    foreach (string file in Directory.EnumerateFiles(folder, "*.jpg"))
//    {
//        Mat img = Cv2.ImRead(file, ImreadModes.Color);
//        Mat imgGray = new Mat();
//        Cv2.CvtColor(img, imgGray, ColorConversionCodes.BGR2GRAY);
//        Mat imgThres = new Mat();
//        Cv2.Threshold(imgGray, imgThres, 150, 250, ThresholdTypes.Otsu);
//        Mat imgResize = new Mat();
//        Cv2.Resize(imgThres, imgResize, size, interpolation: InterpolationFlags.Area);
//        imgResize.GetArray(out byte[] plainArray);
//        NDarray imgNDArray = np.array(plainArray, dtype: np.uint8).reshape(img_size, img_size);
//        test_x_pre.add(imgNDArray);
//        test_y_pre.add(label_y);
//    }
//}

//NDarray test_x = np.array(test_x_pre);
//NDarray test_y = np.array(test_y_pre);
//test_x = test_x.reshape(test_x.shape[0], img_size, img_size, 1);
//test_x = test_x.astype(np.float32);
//test_y = Util.ToCategorical(test_y, num_classes);

//Console.WriteLine($"{train_x.shape[0]} train samples");
//Console.WriteLine($"{test_x.shape[0]} test samples");

//// Build CNN model
//var model = new Sequential();

//model.Add(new Conv2D(32, kernel_size: (3, 3).ToTuple(),
//                        activation: "relu",
//                        input_shape: input_shape));
//model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu"));
//model.Add(new MaxPooling2D(pool_size: (2, 2).ToTuple()));
//model.Add(new Dropout(0.25));
//model.Add(new Flatten());
//model.Add(new Dense(128, activation: "relu"));
//model.Add(new Dropout(0.5));
//model.Add(new Dense(num_classes, activation: "softmax"));

//model.Compile(loss: "categorical_crossentropy",
//    optimizer: new Adadelta(), metrics: new string[] { "accuracy" });

//model.Summary();

//model.Fit(train_x, train_y,
//            batch_size: batch_size,
//            epochs: epochs,
//            verbose: 1
//            //validation_data: new NDarray[] { test_x, test_y }
//            );

////================================> Train loss: 0.0285
////================================> Train accuracy: 0.9923

//var score = model.Evaluate(test_x, test_y, verbose: 0);
//Console.WriteLine($"Test loss: {score[0]}");
//Console.WriteLine($"Test accuracy: {score[1]}");

////================================> Test loss: 0.041980743408203125
////================================> Test accuracy: 0.9828571677207947

////Save model and weights
//string json = model.ToJson();
//File.WriteAllText(modelJsonPath, json);
//model.SaveWeight(modelWeightPath);

//Load model and weight
var loaded_model = Sequential.ModelFromJson(File.ReadAllText(modelJsonPath));
loaded_model.LoadWeight(modelWeightPath);

// ============================================ Predict
Console.WriteLine("==========================>>>> PREDICT ");
string fileTestPath = @"D:\0.Projects\CSS_ImageClassification-master\CSS_ImageClassification\Uploadtemp\a (282).jpg";
List<NDarray> test_predict = new List<NDarray>();
Mat imgtest = Cv2.ImRead(fileTestPath, ImreadModes.Color);
Mat imgGraytest = new Mat();
Cv2.CvtColor(imgtest, imgGraytest, ColorConversionCodes.BGR2GRAY);

Mat imgThresTest = new Mat();
Cv2.Threshold(imgGraytest, imgThresTest, 150, 250, ThresholdTypes.Otsu);
Mat imgResizeTest = new Mat();
Cv2.Resize(imgThresTest, imgResizeTest, size, interpolation: InterpolationFlags.Area);
imgResizeTest.GetArray(out byte[] plainArrayTest);
NDarray imgNDArrayTest = np.array(plainArrayTest, dtype: np.uint8).reshape(img_size, img_size);
test_predict.add(imgNDArrayTest);

NDarray test_predict_x = np.array(test_predict);
test_predict_x = test_predict_x.reshape(test_predict_x.shape[0], img_size, img_size, 1);
test_predict_x = test_predict_x.astype(np.float32);
var predict = loaded_model.Predict(test_predict_x);
int label = int.Parse(string.Join("", np.argmax(predict)));
var percent = np.max(predict) * 100;
Console.WriteLine($"Predicted Label : {(label > 31 ? label - 31 : label)}");
Console.WriteLine($"Predicted Percent : {percent} %");

// code cũ khi dự đoán bằng model đc train bởi ML.NET
//using CSS_ImageClassification;

//var sampleData = new ImageClassification.ModelInput()
//{
//    ImageSource = @"D:\0.Projects\Z_PyThon\LTQG\DuLieuBaoCao\Train\Clean\1\Day_FileTest_1097.jpg",
//};

////Load model and predict output
//var result = ImageClassification.Predict(sampleData);
//Console.WriteLine(" (Predicted Class : " + result.Prediction + ")");
