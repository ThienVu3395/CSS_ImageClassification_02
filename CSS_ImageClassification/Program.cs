////Load sample data
//using CSS_ImageClassification;
//using Microsoft.ML;
using OpenCvSharp;
using OpenCvSharp.Extensions;
//using Tensorflow.Keras.Engine;
//using Tensorflow.Keras.Layers;
//using static Tensorflow.KerasApi;
//using static Tensorflow.Binding;
using Keras;
using Keras.Models;
using Keras.Layers;
using Keras.Optimizers;
using Keras.Utils;
using Numpy;
using Tensorflow;
//using Tensorflow.Layers;
//using System;
//using System.IO;
//using System.Text;
using Keras.Datasets;
using System;
using Python.Runtime;

////Load model and predict output

////string folderPath = @"F:\Projects\PythonFile\LTQG\DuLieuBaoCao\TestClassifier\Clean\";
//string folderPath = @"D:\0.Projects\Z_PyThon\LTQG\DuLieuBaoCao\TestClassifier\Clean\";
//foreach (string folder in Directory.EnumerateFiles(folderPath, "*"))
//{
//    Mat img = Cv2.ImRead(folder);
//    Mat imgGray = new Mat();
//    Cv2.CvtColor(img, imgGray, ColorConversionCodes.BGR2GRAY);
//    Mat imgThres = new Mat();
//    Cv2.Threshold(imgGray, imgThres, 150, 250, ThresholdTypes.Otsu);
//    Mat imgResize = new Mat();
//    Cv2.Resize(imgThres, imgResize, size);
//    //Console.WriteLine(img.GetType());
//    //Console.WriteLine(BitmapConverter.ToBitmap(img).GetType());
//    //Cv2.ImWrite(@"F:\Projects\PythonFile\LTQG\DuLieuBaoCao\TestClassifier\temp.jpg", imgResize);
//    Cv2.ImWrite(@"D:\0.Projects\Z_PyThon\LTQG\DuLieuBaoCao\TestClassifier\temp.jpg", imgResize);

//    var sampleData = new ImageClassification.ModelInput()
//    {
//        //ImageSource = @"F:\Projects\PythonFile\LTQG\DuLieuBaoCao\TestClassifier\temp.jpg",
//        ImageSource = @"D:\0.Projects\Z_PyThon\LTQG\DuLieuBaoCao\TestClassifier\temp.jpg",
//    };

//    var result = ImageClassification.Predict(sampleData);
//    var predictLabel = Int32.Parse(result.Prediction);
//    if (predictLabel > 31)
//    {
//        predictLabel = predictLabel - 31;
//    }
//    Console.WriteLine(folder + " (Predicted Class : " + predictLabel + ")");
//    Console.WriteLine("---------------------------------------------------");
//}

//var model = new Sequential();

//float rate = 0.8F;

//int numClass = 10;

//var inputs = keras.Input(shape: (28, 28, 1), name: "input");

//var layers = new LayersApi();

//var x = layers.Conv2D(32, 3, activation: "relu").Apply(inputs);
//var block_1_output = layers.MaxPooling2D(2).Apply(x);

//x = layers.Conv2D(64, 3, activation: "relu").Apply(block_1_output);
//var block_2_output = layers.MaxPooling2D(2).Apply(x);

//x = layers.Conv2D(32, 3, activation: "relu").Apply(block_2_output);
//var block_3_output = layers.MaxPooling2D(2).Apply(x);

//x = layers.Conv2D(64, 3, activation: "relu").Apply(block_3_output);
//var block_4_output = layers.MaxPooling2D(2).Apply(x);

//x = layers.Conv2D(32, 3, activation: "relu").Apply(block_4_output);
//var block_5_output = layers.MaxPooling2D(2).Apply(x);

//x = layers.Conv2D(64, 3, activation: "relu").Apply(block_5_output);
//var block_6_output = layers.MaxPooling2D(2).Apply(x);

//x = layers.Dense(1024, activation: keras.activations.Relu).Apply(block_6_output);

//x = layers.Dropout(rate).Apply(x);

//var outputs = layers.Dense(numClass, activation: "softmax").Apply(x);

//model = keras.Model(inputs, outputs, name: "output");

//model.summary();

//model.compile(loss: keras.losses.CategoricalCrossentropy(),
//    optimizer: keras.optimizers.Adam(),
//    metrics: new[] { "acc" });

//// prepare dataset
//var ((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data();

//// training
//model.fit(x_train[new Slice(0, 1000)], y_train[new Slice(0, 1000)],
//          batch_size: 64,
//          epochs: 10,
//          validation_split: 0.2f);

int batch_size = 128;
int num_classes = 10;
int epochs = 1;
int img_rows = 28;
int img_cols = 28;
Shape input_shape = null;
Size size = new Size(28, 28);
string folderPath = @"D:\0.Projects\CSS_ImageClassification-master\CSS_ImageClassification\Test\";

// the data, split between train and test sets
var ((x_train, y_train), (x_test, y_test)) = MNIST.LoadData();
Console.WriteLine(x_train[0]);
Console.WriteLine($"x_train shape: {x_train[0].shape}");
Console.WriteLine($"x_train type: {x_train[0].GetType()}");

Mat img = Cv2.ImRead(folderPath + "temp2.jpg", ImreadModes.Color);
Mat imgGray = new Mat();
Cv2.CvtColor(img, imgGray, ColorConversionCodes.BGR2GRAY);
Mat imgThres = new Mat();
Cv2.Threshold(imgGray, imgThres, 150, 250, ThresholdTypes.Otsu);
Mat imgResize = new Mat();
Cv2.Resize(imgThres, imgResize, size);
Console.WriteLine("==========================");
Console.WriteLine(imgResize);
Console.WriteLine($"imgResize size : {imgResize.Size()}");
Console.WriteLine($"imgResize type : {imgResize.GetType()}");

//if (Backend.ImageDataFormat() == "channels_first")
//{
//    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols);
//    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols);
//    input_shape = (1, img_rows, img_cols);
//}
//else
//{
//    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1);
//    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1);
//    input_shape = (img_rows, img_cols, 1);
//}

//x_train = x_train.astype(np.float32);
//x_test = x_test.astype(np.float32);
//x_train /= 255;
//x_test /= 255;
//Console.WriteLine($"x_train shape: {x_train.shape}");
//Console.WriteLine($"{x_train.shape[0]} train samples");
//Console.WriteLine($"{x_test.shape[0]} test samples");

//// convert class vectors to binary class matrices
//y_train = Util.ToCategorical(y_train, num_classes);
//y_test = Util.ToCategorical(y_test, num_classes);

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

//model.Fit(x_train, y_train,
//            batch_size: batch_size,
//            epochs: 1,
//            verbose: 1,
//            validation_data: new NDarray[] { x_test, y_test });

//var score = model.Evaluate(x_test, y_test, verbose: 0);
//Console.WriteLine($"Test loss: {score[0]}");
//Console.WriteLine($"Test accuracy: {score[1]}");

//Mat img = Cv2.ImRead(folderPath);
//Mat imgGray = new Mat();
//Cv2.CvtColor(img, imgGray, ColorConversionCodes.BGR2GRAY);
//Mat imgThres = new Mat();
//Cv2.Threshold(imgGray, imgThres, 150, 250, ThresholdTypes.Otsu);
//Mat imgResize = new Mat();
//Cv2.Resize(imgThres, imgResize, size);

//Load sample data
//using CSS_ImageClassification;

//var sampleData = new ImageClassification.ModelInput()
//{
//    ImageSource = @"D:\0.Projects\Z_PyThon\LTQG\DuLieuBaoCao\Train\Clean\1\Day_FileTest_1097.jpg",
//};

////Load model and predict output
//var result = ImageClassification.Predict(sampleData);
//Console.WriteLine(" (Predicted Class : " + result.Prediction + ")");