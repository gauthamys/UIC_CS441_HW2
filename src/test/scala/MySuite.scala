import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.Encodings
import com.typesafe.config.ConfigFactory
import org.apache.spark.{SparkConf, SparkContext}
import util.{EmbeddingUtil, StrUtil}

import scala.io.Source

class MySuite extends munit.FunSuite {
  private val conf = ConfigFactory.load()
  private val windowSize = conf.getInt("SlidingWindow.windowSize")
  private val embeddingSize = conf.getInt("Embedding.dimension")
  private val registry = Encodings.newDefaultEncodingRegistry()
  private val encoding = registry.getEncoding(EncodingType.CL100K_BASE)

  test("converting array to embedding"){
    val input = "[1, 2, 4, 5, 6, 7]"
    val actual = EmbeddingUtil.convertToArr(input)
    val expected = Array(1D, 2D, 4D, 5D, 6D, 7D)
    assert(expected.sameElements(actual))
  }
  test("sliding window shape test") {
    val sparkConf = new SparkConf().setAppName("test").setMaster("local[2]").set("spark.executor.memory", "6g")
    val sc = new SparkContext(sparkConf)
    val lookup = EmbeddingUtil.loadEmbeddings("src/main/resources/embeddings.txt", sc).value
    val file = Source.fromFile("src/main/resources/ulyss12-sharded.txt")
    val line = file.getLines().toList.head
    file.close()
    val tokens = StrUtil.cleanLine(line)
    val windows = slidingwindow.SlidingWindow.createSlidingWindowsWithPositionalEmbedding(tokens, lookup)
    assert(windows.nonEmpty)
    sc.stop()
    windows.foreach(window => {
      assert(window.getFeatures.shape().sameElements(Array(1L, windowSize.toDouble, 100L)))
      assert(window.getLabels.shape().sameElements(Array(1L, 1L, 100L)))
    })
  }
  test("StrUtil clean test") {
    val input = "This is an,,, unclean!$@&#6 sentence_--0-123;';."
    val actual = StrUtil.cleanLine(input)
    val expected = Array("this", "is", "an", "unclean", "sentence").map(word => encoding.encode(word).toArray.mkString(":"))
    for(i <- actual.indices) {
      assertEquals(actual(i), expected(i))
    }
  }
  test("embedding loading test") {
    val sparkConf = new SparkConf().setAppName("test").setMaster("local[2]").set("spark.executor.memory", "6g")
    val sc = new SparkContext(sparkConf)
    val lookup = EmbeddingUtil.loadEmbeddings("src/main/resources/embeddings.txt", sc).value
    sc.stop()
    assert(!lookup.isEmpty)
    assert(lookup.size() > 0)
  }
  test("word not in vocabulary test") {
    val sparkConf = new SparkConf().setAppName("test").setMaster("local[2]").set("spark.executor.memory", "6g")
    val sc = new SparkContext(sparkConf)
    val lookup = EmbeddingUtil.loadEmbeddings("src/main/resources/embeddings.txt", sc).value
    val file = Source.fromFile("src/main/resources/ulyss12-sharded.txt")
    val line = file.getLines().toList.head
    file.close()
    val tokens = StrUtil.cleanLine(line)
    tokens :+ "UNK"
    val windows = slidingwindow.SlidingWindow.createSlidingWindowsWithPositionalEmbedding(tokens, lookup)
    assert(windows.nonEmpty)
    val lastWindow = windows.tail
    sc.stop()
    assert(!lastWindow.head.getLabels.isEmpty)
  }
}
