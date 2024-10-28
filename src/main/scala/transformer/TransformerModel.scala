package transformer

import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, OutputLayer, RnnOutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

object TransformerModel extends java.io.Serializable{
  def load(): MultiLayerConfiguration = {
    val inputSize = 300 // Feature size for each time step
    val hiddenLayerSize = 64 // Hidden layer size
    val outputSize = 100 // Output size for each time step

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
      // Second LSTM layer
      .layer(1, new LSTM.Builder()
        .nIn(hiddenLayerSize)
        .nOut(hiddenLayerSize)
        .activation(Activation.TANH) // Activation for LSTM
        .build())
      // RNN Output layer: use the output from the last time step
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .nIn(hiddenLayerSize)
        .nOut(outputSize) // Output size for the last time step
        .activation(Activation.IDENTITY) // Identity activation for regression
        .build())
      .build()
  }
}
