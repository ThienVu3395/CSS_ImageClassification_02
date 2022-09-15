﻿
// This file was auto-generated by ML.NET Model Builder. 

using System;

namespace ImageClassification_ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create single instance of sample data from first line of dataset for model input
            ImageClassification.ModelInput sampleData = new ImageClassification.ModelInput()
            {
                ImageSource = @"F:\Projects\PythonFile\LTQG\DuLieuBaoCao\Train\Clean_MLNet\1\Day_FileTest_1097.jpg",
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = ImageClassification.Predict(sampleData);

            Console.WriteLine("Using model to make single prediction -- Comparing actual Label with predicted Label from sample data...\n\n");


            Console.WriteLine($"ImageSource: {@"F:\Projects\PythonFile\LTQG\DuLieuBaoCao\Train\Clean_MLNet\1\Day_FileTest_1097.jpg"}");


            Console.WriteLine($"\n\nPredicted Label value: {predictionResult.Prediction} \nPredicted Label scores: [{String.Join(",", predictionResult.Score)}]\n\n");
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
    }
}
