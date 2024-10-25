package transformer


import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions


object TransformerModel extends java.io.Serializable{
  def createTransformerModel(): MultiLayerConfiguration = {
    val numInputs = 1 // 300x100 flattened input
    val hiddenLayerSize = 1  // Hidden layer size
    val numOutputs = 1      // Output size is 1x100

    val modelConfig: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(123) // Set a seed for reproducibility
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(numInputs)
        .nOut(hiddenLayerSize)
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .nIn(hiddenLayerSize)
        .nOut(numOutputs)
        .activation(Activation.IDENTITY)
        .weightInit(WeightInit.XAVIER)
        .build())
      .build()
    modelConfig
  }
}
