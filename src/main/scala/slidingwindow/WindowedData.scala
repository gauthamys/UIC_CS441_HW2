package slidingwindow

class WindowedData(val input: Array[String], val target: String) extends java.io.Serializable{
  override def toString: String =  {
    input.mkString("[", ",", "]") + ":" + target
  }
}
