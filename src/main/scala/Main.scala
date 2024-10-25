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

object Main {
  private val slidingWindow = SlidingWindow
  private val strUtil = new StrUtil()

  private def createRDDFromData(data: List[DataSet], sc: SparkContext): RDD[DataSet] = {
    // Parallelize your data into a distributed RDD
    sc.parallelize(data)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("llm-training").setMaster("local[2]").set("spark.executor.memory", "6g")
    val sc = new SparkContext(conf)
    val inputPath = "src/main/resources/ulyss12-sharded.txt"
    val sentences = sc.textFile(inputPath)

    val slidingWindows = sentences
      .flatMap(sentence => {
        val clean = strUtil.cleanLine(sentence)
        slidingWindow.createSlidingWindowsWithPositionalEmbedding(clean.split(" "))
      })
      .collect()
      .toList
    val slidingWindowRDD = createRDDFromData(slidingWindows, sc)

    // slidingWindows is a List[DataSet] which can be used to train the LLM
    // Output the number of sliding windows created
    println(s"Number of sliding windows with positional embeddings: ${slidingWindows.size}")

    // Create the Transformer model configuration
    val transformerConfig = TransformerModel.createTransformerModel()
    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(1)  // Batch size per worker
      .averagingFrequency(5)   // Synchronize every 5 iterations
      .workerPrefetchNumBatches(1)  // Prefetch batches to improve performance
      .batchSizePerWorker(1)   // Batch size used by each Spark worker
      .build()
    println("training master built")
    // Wrap the model configuration with SparkDl4jMultiLayer for distributed training
    val sparkModel = new SparkDl4jMultiLayer(sc, transformerConfig, trainingMaster)

    // Set a listener to print score every 10 iterations
    sparkModel.setListeners(new ScoreIterationListener(10))

    // Train the model on the RDD
    val numEpochs = 5
    for(_ <- 0 until numEpochs) {
      println("the problem starts here")
      sparkModel.fit(slidingWindowRDD)
    }
    println("Training complete.")

    sc.stop()
  }
}
