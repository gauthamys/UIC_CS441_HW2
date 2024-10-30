import com.typesafe.config.ConfigFactory

class MySuite extends munit.FunSuite {
  private val conf = ConfigFactory.load()
  private val windowSize = conf.getInt("SlidingWindow.windowSize")

  test("converting array to embedding"){
    true
  }
  test("sliding window shape test") {
    true
  }
  test("StrUtil clean test") {
    true
  }
  test("embedding loading test") {
    true
  }
  test("local training test") {
    true
  }
}
