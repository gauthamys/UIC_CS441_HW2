package util

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import java.util

object EmbeddingUtil extends java.io.Serializable{
  private val registry = Encodings.newDefaultEncodingRegistry()
  private val encoding = registry.getEncoding(EncodingType.CL100K_BASE)

  def convertToArr(v: String): Array[Double] = {
    v.stripPrefix("[").stripSuffix("]").split(",").map(_.trim.toDouble)
  }
  def loadEmbeddings(embeddingPath: String, sc: SparkContext): Broadcast[util.HashMap[String, Array[Double]]] = {
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
}
