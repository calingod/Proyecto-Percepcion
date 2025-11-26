from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col
import os

# ------------------------------
# Configuración de Spark
# ------------------------------
spark = SparkSession.builder \
    .appName("PreprocesamientoCarne") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Ruta base de tu proyecto
base_dir = "/home/calingod/Escritorio/percep/ProyectoCarnesV2/data"

fresh_path = os.path.join(base_dir, "Fresh")
spoiled_path = os.path.join(base_dir, "Spoiled")

# ------------------------------
# Verificar existencia de carpetas
# ------------------------------
if not os.path.exists(fresh_path):
    raise FileNotFoundError(f"La carpeta 'Fresh' no existe en {fresh_path}")

if not os.path.exists(spoiled_path):
    raise FileNotFoundError(f"La carpeta 'Spoiled' no existe en {spoiled_path}")

print("📁 Leyendo imágenes con Spark...")

# ------------------------------
# Leer imágenes
# ------------------------------
df_fresh = spark.read.format("image").load(fresh_path).withColumn("label", lit(0))
df_spoiled = spark.read.format("image").load(spoiled_path).withColumn("label", lit(1))

# ------------------------------
# Unir datasets
# ------------------------------
df = df_fresh.union(df_spoiled)

# ------------------------------
# Filtrar imágenes corruptas
# ------------------------------
df = df.filter(col("image").isNotNull())

# ------------------------------
# Seleccionar solo la ruta y la etiqueta
# ------------------------------
df_light = df.select(col("image.origin").alias("path"), "label")

# ------------------------------
# Reducir particiones para evitar problemas de memoria
# ------------------------------
df_light = df_light.repartition(2)

# ------------------------------
# Mostrar información
# ------------------------------
print("\n📊 Total de imágenes cargadas:", df_light.count())
df_light.show(10, truncate=False)

# ------------------------------
# Guardar como Parquet
# ------------------------------
output_path = "/home/calingod/Escritorio/percep/data_spark_clean"

try:
    df_light.write.mode("overwrite").parquet(output_path)
    print(f"\n💾 Datos procesados guardados en: {output_path}")
except Exception as e:
    print("❌ Error al guardar Parquet:", e)

# ------------------------------
# Finalizar Spark
# ------------------------------
spark.stop()
print("\n🟢 Spark finalizado correctamente.")
