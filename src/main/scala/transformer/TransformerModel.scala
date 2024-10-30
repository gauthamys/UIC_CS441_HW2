package transformer

import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object TransformerModel extends java.io.Serializable{
  def load(): MultiLayerConfiguration = {
    val inputSize = 300 // Feature size for each time step
    val hiddenLayerSize = 100 // Hidden layer size
    val outputSize = 1 // Output size for each time step

    new NeuralNetConfiguration.Builder()
      .seed(123) // For reproducibility
      .weightInit(WeightInit.XAVIER) // Xavier weight initialization
      .list()
      // LSTM layer for sequence processing
      .layer(0, new LSTM.Builder()
        .nIn(inputSize) // Feature size for each time step
        .nOut(hiddenLayerSize) // Hidden layer size
        .activation(Activation.TANH) // Activation for LSTM
        .build())

      // RNN Output layer: use the output from the last time step
      .layer(1, new RnnOutputLayer.Builder()
        .lossFunction(LossFunction.MSE)
        .activation(Activation.IDENTITY) // Identity activation for regression
        .nIn(hiddenLayerSize)
        .nOut(outputSize) // Output size for the last time step
        .build())
      .build()
  }
}
