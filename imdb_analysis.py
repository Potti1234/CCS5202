import os
import sys
from time import time
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col, split, explode
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import matplotlib.pyplot as plt
import seaborn as sns

class IMDBMovieAnalysis:
    def __init__(self):
        # Initialize Spark Session
        self.spark = SparkSession.builder \
            .appName("IMDBMovieAnalysis") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
        
        # Enable more verbose logging
        self.spark.sparkContext.setLogLevel("INFO")
        
        # Define schema for the IMDB dataset
        self.schema = StructType([
            StructField("id", StringType(), True),
            StructField("title", StringType(), True),
            StructField("type", StringType(), True),
            StructField("genres", StringType(), True),
            StructField("averageRating", DoubleType(), True),
            StructField("numVotes", IntegerType(), True),
            StructField("releaseYear", IntegerType(), True)
        ])

    def load_and_preprocess_data(self):
        print("\nLoading and preprocessing data...")
        # Load the dataset
        df = self.spark.read.csv('data.csv', header=True, schema=self.schema)
        
        # Remove any rows with null values
        df_cleaned = df.dropna()
        
        # Explode genres (convert comma-separated genres to array)
        df_cleaned = df_cleaned.withColumn("genres_array", split(col("genres"), ","))
        
        # Create features for clustering
        assembler = VectorAssembler(
            inputCols=["averageRating", "numVotes", "releaseYear"],
            outputCol="features"
        )
        df_features = assembler.transform(df_cleaned)
        
        # Scale features
        scaler = MinMaxScaler(
            inputCol="features",
            outputCol="scaledFeatures"
        )
        scaler_model = scaler.fit(df_features)
        self.df_processed = scaler_model.transform(df_features)
        self.df_processed.cache()
        
        # Show basic statistics
        print("\nDataset Overview:")
        self.df_processed.describe().show()
        
        print("\nMovies by Type:")
        self.df_processed.groupBy("type").count().show()
        
        print("\nTop 10 Genres:")
        self.df_processed.select(explode("genres_array").alias("genre")) \
            .groupBy("genre") \
            .count() \
            .orderBy(col("count").desc()) \
            .show(10)

    def perform_clustering(self, k=5):
        print("\nPerforming K-Means clustering...")
        kmeans = KMeans(featuresCol="scaledFeatures", k=k)
        model = kmeans.fit(self.df_processed)
        
        # Make predictions
        predictions = model.transform(self.df_processed)
        
        # Evaluate clustering
        evaluator = ClusteringEvaluator(
            predictionCol='prediction',
            featuresCol='scaledFeatures',
            metricName='silhouette'
        )
        silhouette = evaluator.evaluate(predictions)
        
        # Analyze clusters
        cluster_analysis = predictions.select(
            "title", "averageRating", "numVotes", "releaseYear", "prediction"
        ).groupBy("prediction").agg({
            "averageRating": "avg",
            "numVotes": "avg",
            "releaseYear": "avg",
            "*": "count"
        }).orderBy("prediction")
        
        print("\nCluster Analysis:")
        cluster_analysis.show()
        print(f"\nSilhouette Score: {silhouette}")
        
        # Save results for visualization
        self.cluster_results = predictions.select(
            "title", "averageRating", "numVotes", "releaseYear", "prediction"
        ).toPandas()

    def create_visualizations(self):
        print("\nCreating visualizations...")
        plt.style.use('seaborn')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Rating distribution by cluster
        sns.boxplot(data=self.cluster_results, x='prediction', y='averageRating', ax=axes[0,0])
        axes[0,0].set_title('Rating Distribution by Cluster')
        
        # Votes distribution by cluster
        sns.boxplot(data=self.cluster_results, x='prediction', y='numVotes', ax=axes[0,1])
        axes[0,1].set_title('Votes Distribution by Cluster')
        
        # Release year distribution by cluster
        sns.boxplot(data=self.cluster_results, x='prediction', y='releaseYear', ax=axes[1,0])
        axes[1,0].set_title('Release Year Distribution by Cluster')
        
        # Cluster sizes
        cluster_sizes = self.cluster_results['prediction'].value_counts()
        cluster_sizes.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Cluster Sizes')
        
        plt.tight_layout()
        plt.savefig('cluster_analysis.png')
        plt.close()
        print("✓ Visualizations saved as 'cluster_analysis.png'")

    def run_analysis(self):
        try:
            start_time = time()
            print("Starting IMDB Movie Analysis Pipeline...")
            
            self.load_and_preprocess_data()
            self.perform_clustering()
            self.create_visualizations()
            
            execution_time = time() - start_time
            print(f"\nAnalysis Pipeline Completed!")
            print(f"Total execution time: {execution_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise e
        
        finally:
            print("\nCleaning up...")
            self.spark.stop()
            print("✓ Spark session stopped")

def main():
    if not os.path.exists('data.csv'):
        print("Error: data.csv file not found!")
        print("Please ensure your IMDB dataset is named 'data.csv' and is in the current directory.")
        sys.exit(1)
    
    analyzer = IMDBMovieAnalysis()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 