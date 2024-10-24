package transformer

import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

class TransformerModel extends java.io.Serializable{
  def createTransformerModel(inputSize: Int, hiddenSize: Int, outputSize: Int): MultiLayerConfiguration = {
    new NeuralNetConfiguration.Builder()
      .seed(42) // For reproducibility
      .updater(new Adam(1e-3)) // Adam optimizer with learning rate 1e-3
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(inputSize)
        .nOut(hiddenSize)
        .activation(Activation.RELU)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(hiddenSize)
        .nOut(hiddenSize)
        .activation(Activation.RELU)
        .build())
      .layer(2, new OutputLayer.Builder(LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)
        .nIn(hiddenSize)
        .nOut(outputSize)
        .build())
      .build()
  }
}
