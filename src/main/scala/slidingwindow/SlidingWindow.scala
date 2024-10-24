package slidingwindow

import com.typesafe.config.ConfigFactory
import embeddingloader.EmbeddingLoader
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import java.util
import scala.collection.convert.ImplicitConversions.`map AsScala`

object SlidingWindow {
  private val conf = ConfigFactory.load()
  private val windowSize = conf.getInt("SlidingWindow.windowSize")
  private val slideLength = conf.getInt("SlidingWindow.slideLength")
  private val lookup = EmbeddingLoader.load()

  // Dummy method to simulate tokenization and embedding (replace with actual embedding code)
  private def tokenizeAndEmbed(tokens: Array[String]): INDArray = {
    // Assume each word is embedded as a 1x100 vector
    // get embedding from hw1
    val embeddingMatrix = Nd4j.zeros(windowSize, 100)
    for (i <- tokens.indices) {
      if (i < windowSize) {
        val word = tokens(i)
        val embedding = lookup.getOrElse(word, Array.fill(100)(0.0))

        if (embedding.length == 100) {
          embeddingMatrix.putRow(i, Nd4j.create(embedding))
        }
      }
    }
    embeddingMatrix
    Nd4j.rand(tokens.length, 100)
  }

  // Compute sinusoidal positional embeddings for a given window size
  private def computePositionalEmbedding(windowSize: Int): INDArray = {
    val embeddingDim = 100 // Dimensionality of word embeddings
    val positionalEncoding = Nd4j.zeros(windowSize, embeddingDim)

    for (pos <- 0 until windowSize) {
      for (i <- 0 until embeddingDim by 2) {
        val angle = pos / math.pow(10000, (2.0 * i) / embeddingDim)
        positionalEncoding.putScalar(Array(pos, i), math.sin(angle))
        positionalEncoding.putScalar(Array(pos, i + 1), math.cos(angle))
      }
    }
    positionalEncoding
  }

  // Create sliding windows for inputs and targets with positional embeddings
  def createSlidingWindowsWithPositionalEmbedding(tokens: Array[String]): List[DataSet] = {
    val res = (0 until tokens.length - windowSize by slideLength).map { i =>
      // Extract input window (windowSize tokens)
      val inputWindow = tokens.slice(i, i + windowSize)

      // Extract the target token (token right after the window)
      val targetToken = tokens(i + windowSize)

      // Convert input tokens into embeddings
      val inputEmbeddings = tokenizeAndEmbed(inputWindow)

      // Add positional embeddings to word embeddings
      val positionalEmbeddings = computePositionalEmbedding(windowSize)
      val positionAwareEmbedding = inputEmbeddings.add(positionalEmbeddings)

      // Convert the target token into an embedding
      val targetEmbedding = tokenizeAndEmbed(Array(targetToken))

      // Add to dataset
      new DataSet(positionAwareEmbedding, targetEmbedding)
    }.toList
    res
  }
}
