import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.LabelLastTimeStepPreProcessor
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration
import slidingwindow.SlidingWindow
import transformer.TransformerModel
import util.StrUtil

import scala.jdk.CollectionConverters.{collectionAsScalaIterableConverter, iterableAsScalaIterableConverter, seqAsJavaListConverter}

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
        slidingWindow.createSlidingWindowsWithPositionalEmbedding(clean)
      })
      .collect()
      .toList
    //slidingWindows.asJava.forEach(ds => println(ds.getFeatures.shape().mkString("Array(", ", ", ")")))
    // Output the number of sliding windows created
    println(s"Number of sliding windows with positional embeddings: ${slidingWindows.size}")

    // Create the Transformer model configuration
    val transformerConfig = TransformerModel.load()

    //-----------LOCAL-------------
    val model = new MultiLayerNetwork(transformerConfig)
    val iter = new ListDataSetIterator[DataSet](slidingWindows.asJava, 1)
    model.fit(iter)

    //----------SPARK---------------
    // Wrap the model configuration with SparkDl4jMultiLayer for distributed training
//    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(2)  // Batch size per worker
//      .rngSeed(123)
//      .collectTrainingStats(true)
//      .batchSizePerWorker(1)
//      .build()
//    val sparkModel = new SparkDl4jMultiLayer(sc, transformerConfig, trainingMaster)
//
//    // Set a listener to print score every 10 iterations
//    sparkModel.setListeners(new ScoreIterationListener(10))
//
//    // Train the model on the RDD
//    val slidingWindowRDD = createRDDFromData(slidingWindows, sc)
//    println(slidingWindowRDD.count())
//    val numEpochs = 5
//    for (_ <- 0 until numEpochs){
//      println("the problem starts here")
//      sparkModel.fit(slidingWindowRDD)
//    }
//    println("Training complete.")

    sc.stop()
  }
}
