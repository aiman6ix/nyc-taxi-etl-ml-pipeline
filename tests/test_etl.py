import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder
        .master("local[1]")
        .appName("ci-pyspark-test")
        .getOrCreate()
    )


def test_spark_session_runs(spark):
    data = [("NYC", 1)]
    df = spark.createDataFrame(data, ["city", "id"])
    assert df.count() == 1