import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract, col

HDFS_URI = os.environ.get("HDFS_URI", "hdfs://namenode:9000")
SPARK_MASTER = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")

def main():
    spark = (
        SparkSession.builder
        .appName("MeatQualityPreprocess")
        .master("local[*]")   # Spark local dentro del contenedor
        .getOrCreate()
    )

    hdfs_path = f"{HDFS_URI}/meat_quality/train/*/*"
    print(f"Leyendo imágenes desde: {hdfs_path}")

    df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(hdfs_path)
        .withColumn("filename", input_file_name())
    )

    # Extraemos la etiqueta (Fresh/Spoiled) de la ruta
    # asumiendo estructura /meat_quality/train/<label>/nombre.jpg
    df = df.withColumn(
        "label",
        regexp_extract(col("filename"), r"/train/([^/]+)/", 1)
    )

    df.show(5, truncate=False)

    counts = df.groupBy("label").count()
    print("Conteo por clase en HDFS:")
    counts.show()

    out_path = f"{HDFS_URI}/meat_quality/summary_counts"
    print(f"Guardando resumen en: {out_path}")
    counts.write.mode("overwrite").parquet(out_path)

    spark.stop()


if __name__ == "__main__":
    main()
