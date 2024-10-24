
val scala3Version = "3.5.1"
val scala2Version = "2.12.18"

lazy val root = project
  .in(file("."))
  .settings(
    name := "hw2",
    version := "0.1.0-SNAPSHOT",

    scalaVersion := scala2Version,

    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "1.0.0" % Test,
      "com.typesafe" % "config" % "1.4.3",
      "com.knuddels" % "jtokkit" % "1.1.0",

      ("org.apache.spark" %% "spark-sql" % "3.5.3").cross(CrossVersion.for3Use2_13),
      ("org.apache.spark" %% "spark-core" % "3.5.3").cross(CrossVersion.for3Use2_13),
      ("org.apache.spark" %% "spark-mllib" % "3.5.3").cross(CrossVersion.for3Use2_13),

      "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1",
      "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
      "org.deeplearning4j" %% "dl4j-spark" % "1.0.0-M2.1",

      "org.nd4j" % "nd4j-native" % "1.0.0-M2.1",
      "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1",
    ),
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", _ @ xs_*) => MergeStrategy.discard
      case PathList("META-INF", "services", _ @ xs_*) => MergeStrategy.concat
      case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
      case "reference.conf" => MergeStrategy.concat
      case _ => MergeStrategy.first
    },
    assembly / assemblyJarName := "hw2.jar"
  )
