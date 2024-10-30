import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.deeplearning4j.optimize.listeners.{PerformanceListener, ScoreIterationListener}
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import slidingwindow.SlidingWindow
import transformer.TransformerModel
import util.StrUtil
import util.EmbeddingUtil

import java.util
import scala.jdk.CollectionConverters.{collectionAsScalaIterableConverter, seqAsJavaListConverter}

object Main {
  private val slidingWindow = SlidingWindow
  private val conf = ConfigFactory.load()
  private val batchSize = conf.getInt("Training.batchSize")
  private val numEpochs = conf.getInt("Training.numEpochs")
  private val registry = Encodings.newDefaultEncodingRegistry()
  private val encoding = registry.getEncoding(EncodingType.CL100K_BASE)

  private def createRDDFromData(data: List[DataSet], sc: SparkContext): RDD[DataSet] = {
    // Parallelize your data into a distributed RDD
    sc.parallelize(data)
  }

  private def loadEmbeddings(embeddingPath: String, sc: SparkContext): Broadcast[util.HashMap[String, Array[Double]]] = {
    val lookup = new util.HashMap[String, Array[Double]]()
    val wordEmbeddingsRDD = sc.textFile(embeddingPath)
    val wordEmbeddings = wordEmbeddingsRDD
      .map(_.split("\t"))
      .map{ kv =>
        val word = kv(0)
        val embedding = EmbeddingUtil.convertToArr(kv(1))
        (encoding.encode(word).toArray.mkString(":"), embedding)
      }
    wordEmbeddings.collect().foreach { case (word, embedding) =>
      lookup.put(word, embedding)
    }
    sc.broadcast(lookup)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("llm-training").setMaster("local[2]").set("spark.executor.memory", "6g")
    val sc = new SparkContext(conf)

//    val inputPath = args(0)
//    val embeddingPath = args(1)
//    val statsFilePath = args(2)
//    val outputPath = args(3)
//    val sentences = sc.textFile(inputPath)

    val inputPath = "src/main/resources/ulyss12-sharded.txt"
    val embeddingPath = "src/main/resources/embeddings.txt"
    val statsFilePath = "results/training-stats"
    val outputPath = "results/model.zip"

    // embeddings
    val lookup = loadEmbeddings(embeddingPath, sc).value

    // sliding windows
    val sentences = sc.textFile(inputPath)
    val slidingWindows = sentences
      .flatMap(sentence => {
        val clean = StrUtil.cleanLine(sentence)
        slidingWindow.createSlidingWindowsWithPositionalEmbedding(clean, lookup)
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
    sparkModel.setListeners(new PerformanceListener(1))
    sparkModel.setCollectTrainingStats(true)

    // Train the model on the RDD
    val slidingWindowRDD = createRDDFromData(batched, sc)
    println(slidingWindowRDD.count())

    for (_ <- 0 until numEpochs){
      val start = System.currentTimeMillis()
      sparkModel.fit(slidingWindowRDD)
      val end = System.currentTimeMillis()
      println(s"Epoch time: ${end - start}ms")
    }
    println("Training complete.")

    val network = sparkModel.getNetwork
    ModelSerializer.writeModel(network, outputPath, true)
    sparkModel.getSparkTrainingStats.exportStatFiles(statsFilePath, sc)

    sc.stop()
  }
}
