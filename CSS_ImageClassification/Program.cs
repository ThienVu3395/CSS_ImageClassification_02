using CSS_ImageClassification;
using Microsoft.ML;
using OpenCvSharp;
using Keras;
using Keras.Models;
using Keras.Layers;
using Keras.Optimizers;
using Keras.Utils;
using Numpy;
using Tensorflow;
using Python.Included;
using Python.Runtime;
using System.Threading;
using System.Linq;

// Chạy nếu báo thiếu DLL "pythonXX"
await Installer.SetupPython();
//Installer.TryInstallPip(); // chạy lần đầu, sau có thể comment lại
//Installer.PipInstallModule("tensorflow"); // chạy lần đầu, sau có thể comment lại
PythonEngine.Initialize();
//dynamic tf = Py.Import("tensorflow"); // chạy lần đầu, sau có thể comment lại
//Console.WriteLine("TensorFlow version : " + tf.__version__); // chạy lần đầu, sau có thể comment lại

// Khai báo các biến cần thiết
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

////Training Model
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
//        Cv2.Resize(imgThres, imgResize, size, interpolation: InterpolationFlags.Area);
//        imgResize.GetArray(out byte[] plainArray);
//        NDarray imgNDArray = np.array(plainArray, dtype: np.uint8).reshape(img_size, img_size);
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

/////////////// Predict
Console.WriteLine("==========================>>>> PREDICT");
string fileTestPath = @"D:\0.Projects\CSS_ImageClassification-master\CSS_ImageClassification\Uploadtemp\img_7.jpg";
int plusSize = 1;
int crop_number = 0;
List<NDarray> test_predict = new List<NDarray>();

Mat img = Cv2.ImRead(fileTestPath, ImreadModes.Color);

// "Resize image" python code function
Mat imgBorder = img.CopyMakeBorder(plusSize, plusSize, plusSize, plusSize, borderType: BorderTypes.Constant, value: new Scalar(255, 255, 255, 0));
Mat imgResize = imgBorder.Resize(new Size(150, 150));

int height = imgResize.Height;
int width = imgResize.Width;

// "CUT_ROIS" python code function
Mat gray = imgResize.CvtColor(ColorConversionCodes.BGR2GRAY);
Mat thres = gray.Threshold(127, 255, ThresholdTypes.Otsu);
Point[][] points = thres.FindContoursAsArray(RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);

List<Rect> imgCordinates = new List<Rect>();
Mat bigImage = new Mat();
foreach (var item in points)
{
    Rect boundingRect = Cv2.BoundingRect(item);
    int x = boundingRect.X;
    int y = boundingRect.Y;
    int w = boundingRect.Width;
    int h = boundingRect.Height;
    if (w >= width && h >= height)
    {
        bigImage = gray[y, y + h, x, x + w];
    }
    if (w > 10 && w < 130 && h > 30)
    {
        if (crop_number < 8)
        {
            imgCordinates.Add(boundingRect);
            //Cv2.PutText(imgtestGray, h, (x, y + 20), HersheyFonts.HersheySimplex, 1, (36, 255, 12), 2);
            //Cv2.Rectangle(imgtestGray, (x, y), (x + w, y + h), (36,255,12), 1)
            crop_number += 1;
        }
        break;
    }
}

Mat result = new Mat();
int lenImgCor = imgCordinates.Count();
Console.WriteLine($"lenImgCor : {lenImgCor}");

if (lenImgCor == 7)
{
    Rect box1 = imgCordinates[0];
    Rect box2 = imgCordinates[1];
    Rect box3 = imgCordinates[2];
    Rect box4 = imgCordinates[3];
    Rect box5 = imgCordinates[4];
    Rect box6 = imgCordinates[5];
    Rect box7 = imgCordinates[6];
    int[] minleft = { box1.X, box2.X, box3.X, box4.X, box5.X, box6.X, box7.X };
    int minleftResult = minleft.Min();

    int[] minTop = { box1.Y, box2.Y, box3.Y, box4.Y, box5.Y, box6.Y, box7.Y };
    int minTopResult = minTop.Min();

    int[] maxRight = { box1.X + box1.Width, box2.X + box2.Width, box3.X + box3.Width, box4.X + box4.Width, box5.X + box5.Width, box6.X + box6.Width, box7.X + box7.Width };
    int maxRightResult = maxRight.Max();

    int[] maxBottom = { box1.Y + box1.Height, box2.Y + box2.Height, box3.Y + box3.Height, box4.Y + box4.Height, box5.Y + box5.Height, box6.Y + box6.Height, box7.Y + box7.Height };
    int maxBottomResult = maxBottom.Max();

    result = gray[minTopResult, maxBottomResult, minleftResult, maxRightResult];
}

if (lenImgCor == 6)
{
    Rect box1 = imgCordinates[0];
    Rect box2 = imgCordinates[1];
    Rect box3 = imgCordinates[2];
    Rect box4 = imgCordinates[3];
    Rect box5 = imgCordinates[4];
    Rect box6 = imgCordinates[5];
    int[] minleft = { box1.X, box2.X, box3.X, box4.X, box5.X, box6.X };
    int minleftResult = minleft.Min();

    int[] minTop = { box1.Y, box2.Y, box3.Y, box4.Y, box5.Y, box6.Y};
    int minTopResult = minTop.Min();

    int[] maxRight = { box1.X + box1.Width, box2.X + box2.Width, box3.X + box3.Width, box4.X + box4.Width, box5.X + box5.Width, box6.X + box6.Width};
    int maxRightResult = maxRight.Max();

    int[] maxBottom = { box1.Y + box1.Height, box2.Y + box2.Height, box3.Y + box3.Height, box4.Y + box4.Height, box5.Y + box5.Height, box6.Y + box6.Height };
    int maxBottomResult = maxBottom.Max();

    result = gray[minTopResult, maxBottomResult, minleftResult, maxRightResult];
}

if (lenImgCor == 5)
{
    Rect box1 = imgCordinates[0];
    Rect box2 = imgCordinates[1];
    Rect box3 = imgCordinates[2];
    Rect box4 = imgCordinates[3];
    Rect box5 = imgCordinates[4];
    int[] minleft = { box1.X, box2.X, box3.X, box4.X, box5.X };
    int minleftResult = minleft.Min();

    int[] minTop = { box1.Y, box2.Y, box3.Y, box4.Y, box5.Y };
    int minTopResult = minTop.Min();

    int[] maxRight = { box1.X + box1.Width, box2.X + box2.Width, box3.X + box3.Width, box4.X + box4.Width, box5.X + box5.Width };
    int maxRightResult = maxRight.Max();

    int[] maxBottom = { box1.Y + box1.Height, box2.Y + box2.Height, box3.Y + box3.Height, box4.Y + box4.Height, box5.Y + box5.Height };
    int maxBottomResult = maxBottom.Max();

    result = gray[minTopResult, maxBottomResult, minleftResult, maxRightResult];
}

if (lenImgCor == 4)
{
    Rect box1 = imgCordinates[0];
    Rect box2 = imgCordinates[1];
    Rect box3 = imgCordinates[2];
    Rect box4 = imgCordinates[3];
    int[] minleft = { box1.X, box2.X, box3.X, box4.X };
    int minleftResult = minleft.Min();

    int[] minTop = { box1.Y, box2.Y, box3.Y, box4.Y};
    int minTopResult = minTop.Min();

    int[] maxRight = { box1.X + box1.Width, box2.X + box2.Width, box3.X + box3.Width, box4.X + box4.Width};
    int maxRightResult = maxRight.Max();

    int[] maxBottom = { box1.Y + box1.Height, box2.Y + box2.Height, box3.Y + box3.Height, box4.Y + box4.Height};
    int maxBottomResult = maxBottom.Max();

    result = gray[minTopResult, maxBottomResult, minleftResult, maxRightResult];
}

if (lenImgCor == 3)
{
    Rect box1 = imgCordinates[0];
    Rect box2 = imgCordinates[1];
    Rect box3 = imgCordinates[2];
    int[] minleft = { box1.X, box2.X, box3.X};
    int minleftResult = minleft.Min();

    int[] minTop = { box1.Y, box2.Y, box3.Y };
    int minTopResult = minTop.Min();

    int[] maxRight = { box1.X + box1.Width, box2.X + box2.Width, box3.X + box3.Width};
    int maxRightResult = maxRight.Max();

    int[] maxBottom = { box1.Y + box1.Height, box2.Y + box2.Height, box3.Y + box3.Height};
    int maxBottomResult = maxBottom.Max();

    result = gray[minTopResult, maxBottomResult, minleftResult, maxRightResult];
}

if (lenImgCor == 2)
{
    Rect box1 = imgCordinates[0];
    Rect box2 = imgCordinates[1];
    int[] minleft = { box1.X, box2.X };
    int minleftResult = minleft.Min();

    int[] minTop = { box1.Y, box2.Y };
    int minTopResult = minTop.Min();

    int[] maxRight = { box1.X + box1.Width, box2.X + box2.Width };
    int maxRightResult = maxRight.Max();

    int[] maxBottom = { box1.Y + box1.Height, box2.Y + box2.Height };
    int maxBottomResult = maxBottom.Max();

    result = gray[minTopResult, maxBottomResult, minleftResult, maxRightResult];
}

if (lenImgCor == 1)
{
    Rect box1 = imgCordinates[0];
    int minleft = box1.X;
    int minTop = box1.Y;
    int maxRight = box1.Width;
    int maxBottom = box1.Height;
    result = gray[minTop, minTop + maxBottom, minleft, minleft + maxRight];
}

Mat finalResult = new Mat();
Cv2.BitwiseNot(result, finalResult);

Cv2.ImWrite(@"D:\0.Projects\CSS_ImageClassification-master\CSS_ImageClassification\Uploadtemp\img_7_test.jpg", finalResult);

Mat thres2 = finalResult.Threshold(150, 250, ThresholdTypes.Otsu);
Mat resize2 = thres2.Resize(size, interpolation: InterpolationFlags.Area);
resize2.GetArray(out byte[] plainArrayTest);
NDarray imgNDArrayTest = np.array(plainArrayTest, dtype: np.uint8).reshape(img_size, img_size);
test_predict.add(imgNDArrayTest);

NDarray test_predict_x = np.array(test_predict);
test_predict_x = test_predict_x.reshape(test_predict_x.shape[0], img_size, img_size, 1);
test_predict_x = test_predict_x.astype(np.float32);
var predict = loaded_model.Predict(test_predict_x);
int labelPredict = int.Parse(string.Join("", np.argmax(predict)));
var percentPredict = np.max(predict) * 100;
Console.WriteLine($"Predicted Label : {(labelPredict > 31 ? labelPredict - 31 : labelPredict)}");
Console.WriteLine($"Predicted Percent : {percentPredict} %");

// code cũ khi dự đoán bằng model đc train bởi ML.NET
//using CSS_ImageClassification;

//var sampleData = new ImageClassification.ModelInput()
//{
//    ImageSource = @"D:\0.Projects\Z_PyThon\LTQG\DuLieuBaoCao\Train\Clean\1\Day_FileTest_1097.jpg",
//};

////Load model and predict output
//var result = ImageClassification.Predict(sampleData);
//Console.WriteLine(" (Predicted Class : " + result.Prediction + ")");
