import sys, time
from pyspark.sql import SparkSession, functions as F

# arguments passed in from the command line
input_path = sys.argv[1]

# (A1) start spark session 
spark = SparkSession.builder.appName("ws5-regression").getOrCreate()
df = spark.read.json(input_path)

