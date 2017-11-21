package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature.{IDFModel, IDF}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.regression.LabeledPoint

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /** 1 - CHARGEMENT DU DATAFRAME **/

    val dataframe = spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("nullValue", "false")
      .parquet("/Users/TheoAndro/Downloads/TP_ParisTech_2017_2018_starter/prepared_trainingset")



    /** A) PREMIER STAGE : ON SEPARE LES TEXTES EN MOTS  **/

    val tokenizer = new RegexTokenizer()
      .setPattern( "\\W+" )
      .setGaps( true )
      .setInputCol( "text" )
      .setOutputCol( "tokens" )



    /** B) DEUXIEME STAGE : ON ENLEVE LES STOP WORDS **/
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered_SW")



    /** C) TROISIEME STAGE : APPELLE DU COUNTVECTORIZER **/

    val count_vect: CountVectorizer/*Model*/ = new CountVectorizer()
      .setInputCol("filtered_SW")
      .setOutputCol("tf-count")//.fit(removed)



    /** D) QUATRIEME STAGE : RECHERCHE DE LA PARTIE IDF ET ECRITURE DE L'OUTPUT DANS UNE COLONNE "TFIDF"**/

    val idf: IDF/*Model*/ = new IDF()
      .setInputCol("tf-count")
      .setOutputCol("TFIDF")


    /** 3 **/

    /** E) CINQUIEME STAGE : CONVERSION DE COUNTRY2 EN COUNTRY2_INDEXED (VALEUR NUMERIQUE) **/
    val indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country2_indexed")



    /** F) SIXIEME STAGE : CONVERSION DE CURRENCY EN CURRENCY_INDEXED (VALEURS NUMERIQUES) **/
    val indexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency2_indexed")


    /** 4 - CONVERSION DES DONNEES  **/

    /** G) SEPTIEME STAGE : VECTOR ASSEMBLER **/
    val vector_assembler = new VectorAssembler()
      .setInputCols(Array("TFIDF", "days_campaign", "hours_prepa", "goal", "country2_indexed", "currency2_indexed"))
      .setOutputCol("Features")


    /** H) HUITIEME STAGE : REGRESSION LOGISTIQUE **/

    val lr = new LogisticRegression()
      .setElasticNetParam( 0.0 )
      .setFitIntercept(true)
      .setFeaturesCol( "Features" )
      .setLabelCol( "final_status" )
      .setStandardization( true )
      .setPredictionCol( "predictions" )
      .setRawPredictionCol( "raw_predictions" )
      .setThresholds( Array ( 0.7 , 0.3 ))
      .setTol( 1.0e-6 )
      .setMaxIter( 300 )


    /** I) PIPELINE **/
    val Pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, count_vect, idf, indexer, indexer2, vector_assembler, lr))


    /** 5  **/

    /** J) FORMATION DU TRAINSET ET TESTSET **/
    val Array(training, test) = dataframe.randomSplit(Array(0.9, 0.1))


    /** K) PREPARATION DU GRID-SEARCH  **/
    val Grid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.00000001, 0.000001, 0.0001, 0.01))
      .addGrid(count_vect.minDF, Array(55.0, 75.0, 95.0))
      .build()

    /**  UTILISATION DU GRID-SEARCH **/

    val f1_Score= new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")


    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(f1_Score)
      .setEstimator(Pipeline)
      .setEstimatorParamMaps(Grid)
      .setTrainRatio(0.7)


    /** L) APPLICATION DU MEILLEUR MODELE AUX DONNEES TEST**/

    val best_model = trainValidationSplit.fit(training)
    val df_WithPredictions = best_model.transform(test)

    print("Resultat obtenu sur le test: " + f1_Score.evaluate(df_WithPredictions))

    /** M) AFFICHAGE DES PREDICTIONS **/
    df_WithPredictions.groupBy( "final_status" , "predictions" ).count.show()
  }
}
