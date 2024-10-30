package util

object EmbeddingUtil extends java.io.Serializable{
  def convertToArr(v: String): Array[Double] = {
    v.stripPrefix("[").stripSuffix("]").split(",").map(_.trim.toDouble)
  }
}
