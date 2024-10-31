import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.deeplearning4j.optimize.listeners.{PerformanceListener, ScoreIterationListener}
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.LoggerFactory
import slidingwindow.SlidingWindow
import transformer.TransformerModel
import util.StrUtil
import util.EmbeddingUtil

import scala.jdk.CollectionConverters.{collectionAsScalaIterableConverter, seqAsJavaListConverter}

object Main {
  private val slidingWindow = SlidingWindow
  private val conf = ConfigFactory.load()
  private val batchSize = conf.getInt("Training.batchSize")
  private val numEpochs = conf.getInt("Training.numEpochs")
  private val logger = LoggerFactory.getLogger(this.getClass.getName)

  private def createRDDFromData(data: List[DataSet], sc: SparkContext): RDD[DataSet] = {
    // Parallelize your data into a distributed RDD
    sc.parallelize(data)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("llm-training").setMaster("local[2]").set("spark.executor.memory", "6g")
    val sc = new SparkContext(conf)

    // FOR EMR
    val inputPath = args(0)
    val embeddingPath = args(1)
    val statsFilePath = args(2)
    val outputPath = args(3)

      // LOCAL EXECUTION
//    val inputPath = "src/main/resources/ulyss12-sharded.txt"
//    val embeddingPath = "src/main/resources/embeddings.txt"
//    val statsFilePath = "results/training-stats"
//    val outputPath = "results/model.zip"

    // embeddings
    logger.info("Loading Embeddings from HW1")
    val lookup = EmbeddingUtil.loadEmbeddings(embeddingPath, sc).value

    // sliding windows
    logger.info("Creating Sliding Windows from sharded input")
    val sentences = sc.textFile(inputPath)
    val slidingWindows = sentences
      .flatMap(sentence => {
        val clean = StrUtil.cleanLine(sentence)
        slidingWindow.createSlidingWindowsWithPositionalEmbedding(clean, lookup)
      })
      .collect()
      .toList
    logger.info("Generated " + slidingWindows.length + " sliding window examples")

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

    logger.info("Spark model initialised")
    val sparkModel = new SparkDl4jMultiLayer(sc, transformerConfig, trainingMaster)

    // Set a listener to print score every 10 iterations
    sparkModel.setListeners(new ScoreIterationListener(10))
    sparkModel.setListeners(new PerformanceListener(1))
    sparkModel.setCollectTrainingStats(true)

    // Train the model on the RDD
    val slidingWindowRDD = createRDDFromData(batched, sc)

    logger.info("Starting training")
    for (i <- 0 until numEpochs){
      val start = System.currentTimeMillis()
      sparkModel.fit(slidingWindowRDD)
      val end = System.currentTimeMillis()
      logger.info(s"Epoch ${i + 1} time: ${end - start}ms")
    }
    logger.info("Training complete.")

    val network = sparkModel.getNetwork
    ModelSerializer.writeModel(network, outputPath, true)
    logger.info(s"Saved model to $outputPath")
    sparkModel.getSparkTrainingStats.exportStatFiles(statsFilePath, sc)
    logger.info(s"Saved stats to $statsFilePath")

    sc.getExecutorMemoryStatus.toArray.foreach(item => {
      logger.info(s"Memory for ${item._1}: ${item._2.toString()}")
    })
    sc.stop()
  }
}
