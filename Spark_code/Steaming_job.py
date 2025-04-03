from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
 
# Initialize Spark
conf = SparkConf().setAppName("RealTimeLearning")
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 2)
 
# Consume from Kafka
kafka_stream = KafkaUtils.createDirectStream(ssc, ["topic_name"], {"metadata.broker.list": "localhost:9092"})
 
# Preprocess data
processed_data = kafka_stream.map(lambda x: preprocess(x))
 
# Save processed data
processed_data.pprint()
 
ssc.start()
ssc.awaitTermination()