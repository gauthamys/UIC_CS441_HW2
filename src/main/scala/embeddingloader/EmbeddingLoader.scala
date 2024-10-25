package embeddingloader

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import org.apache.spark.{SparkConf, SparkContext}

import java.util

object EmbeddingLoader extends java.io.Serializable{
  private val registry = Encodings.newDefaultEncodingRegistry()
  private val encoding = registry.getEncoding(EncodingType.CL100K_BASE)

  private def convertToArr(v: String): Array[Double] = {
    v.stripPrefix("[").stripSuffix("]").split(",").map(_.trim.toDouble)
  }

  def load(): util.HashMap[String, Array[Double]] = {
    val conf = new SparkConf().setAppName("embedding-loader").setMaster("local[*]").set("spark.driver.memory", "6g")
    val sc = new SparkContext(conf)

    val filePath = "src/main/resources/embeddings.txt"
    val lookup = new util.HashMap[String, Array[Double]]()

    // Load the file into an RDD
    val embeddingsRDD = sc.textFile(filePath)

    // Split lines and filter valid ones
    val wordEmbeddings = embeddingsRDD
      .map(_.split("\t"))
      .map{ kv =>
        val word = kv(0)
        val embedding = convertToArr(kv(1))
        (encoding.encode(word).toArray.mkString(":"), embedding)
      }

    // Collect the results and put them in the HashMap
    wordEmbeddings.collect().foreach { case (word, embedding) =>
      lookup.put(word, embedding)
    }
    sc.stop()
    lookup
  }
}
