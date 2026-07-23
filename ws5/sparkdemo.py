import sys, time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# arguments passed in from the command line
input_path = sys.argv[1]

# (A1) start spark session 
spark = SparkSession.builder.appName("ws5-regression").getOrCreate()

# (A2) read dataset from bucket
df = spark.read.csv(input_path, header=True, inferSchema=True)
df.show()

# (A3) Combine the two predictor columns 
vec_assembler = VectorAssembler(inputCols=["total_bill", "size"], outputCol="features")

# (A4) Split the data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# (A5) Define a LinearRegression
lin_reg = LinearRegression(featuresCol="features", labelCol="tip")
pipeline = Pipeline(stages=[vec_assembler, lin_reg])
pipeline_fitted = pipeline.fit(train_df)

# (A6) Apply the fitted pipeline to the test set
predictions = pipeline_fitted.transform(test_df)

# (A7) Evaluate the predictions on two metrics: RMSE and R².
evaluator = RegressionEvaluator(labelCol="tip", predictionCol="prediction")
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

# (A8) 
lin_reg_model = pipeline_fitted.stages[-1]
print(f"Coefficients: {lin_reg_model.coefficients}")
print(f"Intercept: {lin_reg_model.intercept}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

