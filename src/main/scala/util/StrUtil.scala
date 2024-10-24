package util

class StrUtil extends java.io.Serializable{
  def cleanLine(v: String): String = {
    var clean = v.trim.toLowerCase.replaceAll("[,./?_\"{}()~@!#$%^&*:;0-9<>']", "")
    clean = clean.replace("[", "")
    clean = clean.replace("]", "")
    clean = clean.replace("+", "")
    clean = clean.replace("-", "")
    clean
  }
}
