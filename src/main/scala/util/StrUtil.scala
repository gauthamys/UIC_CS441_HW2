package util

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType

class StrUtil extends java.io.Serializable{
  private val registry = Encodings.newDefaultEncodingRegistry()
  private val encoding = registry.getEncoding(EncodingType.CL100K_BASE)

  def cleanLine(v: String): String = {
    var clean = v.trim.toLowerCase.replaceAll("[,./?_\"{}()~@!#$%^&*:;0-9<>']", "")
    clean = clean.replace("[", "")
    clean = clean.replace("]", "")
    clean = clean.replace("+", "")
    clean = clean.replace("-", "")
    clean.split(" ").map(word => encoding.encode(word).toArray.mkString(":")).mkString(" ")
  }
}
