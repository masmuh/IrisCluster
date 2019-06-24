using Microsoft.ML;
using System;
using System.IO;

namespace ML_Iris_Clustering
{
	class Program
	{
		static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
		static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");
		static void Main(string[] args)
		{
			// <SnippetCreateContext>
			var mlContext = new MLContext(seed: 0);

			// <SnippetCreateDataView>
			IDataView dataView = mlContext.Data.LoadFromTextFile<IrisData>(_dataPath, hasHeader: false, separatorChar: ',');

			string featuresColumnName = "Features";
			var pipeline = mlContext.Transforms
				.Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
				.Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

			var model = pipeline.Fit(dataView);

			using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
			{
				mlContext.Model.Save(model, dataView.Schema, fileStream);
			}
			var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);

			IrisData Setosa = new IrisData
			{
				SepalLength = 5.1f,
				SepalWidth = 3.5f,
				PetalLength = 1.4f,
				PetalWidth = 0.2f
			};

			var prediction = predictor.Predict(Setosa);
			Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
			Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
			Console.ReadKey();
		}
	}
}
