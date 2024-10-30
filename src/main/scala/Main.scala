import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.dataset.DataSet
import slidingwindow.SlidingWindow
import transformer.TransformerModel
import util.StrUtil

import scala.jdk.CollectionConverters.{collectionAsScalaIterableConverter, seqAsJavaListConverter}

object Main {
  private val slidingWindow = SlidingWindow
  private val strUtil = new StrUtil()
  private val conf = ConfigFactory.load()
  private val batchSize = conf.getInt("Training.batchSize")
  private val numEpochs = conf.getInt("Training.numEpochs")

  private def createRDDFromData(data: List[DataSet], sc: SparkContext): RDD[DataSet] = {
    // Parallelize your data into a distributed RDD
    sc.parallelize(data)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("llm-training").setMaster("local[2]").set("spark.executor.memory", "6g")
    val sc = new SparkContext(conf)
    val inputPath = "src/main/resources/ulyss12-sharded.txt"
    val sentences = sc.textFile(inputPath)

    // sliding windows
    val slidingWindows = sentences
      .flatMap(sentence => {
        val clean = strUtil.cleanLine(sentence)
        slidingWindow.createSlidingWindowsWithPositionalEmbedding(clean)
      })
      .collect()
      .toList

    // batched data
    val merged = DataSet.merge(slidingWindows.asJava)
    val batched = merged.batchBy(batchSize).asScala.toList

    // Create the Transformer model configuration
    val transformerConfig = TransformerModel.load()

    // Wrap the model configuration with SparkDl4jMultiLayer for distributed training
    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(1)
      .rngSeed(123)
      .collectTrainingStats(true)
      .batchSizePerWorker(1)
      .build()
    val sparkModel = new SparkDl4jMultiLayer(sc, transformerConfig, trainingMaster)

    // Set a listener to print score every 10 iterations
    sparkModel.setListeners(new ScoreIterationListener(10))

    // Train the model on the RDD
    val slidingWindowRDD = createRDDFromData(batched, sc)
    println(slidingWindowRDD.count())

    for (_ <- 0 until numEpochs){
      sparkModel.fit(slidingWindowRDD)
    }
    println("Training complete.")

    sc.stop()
  }
}
