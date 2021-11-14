#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 11:52:53 2021

@author: prokmar
"""
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import Window

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('len(sys.argv) == {}'.format(len(sys.argv)))
        for i in range(len(sys.argv)):
            print('i - {} = {}'.format(i, sys.argv[i]))
        print('Usage: offense_codes.csv, crime.csv and path/to/output_folder', file=sys.stderr)
        sys.exit(-1)
    
    # создаем сессию Spark    
    spark = (SparkSession.builder.appName('spark_lab').getOrCreate())    
    
    offense_codes = sys.argv[1]
    crime = sys.argv[2]
    path_to_folder = sys.argv[3]
    
    # читаем файл offense_codes.csv
    oc_df = (spark.read.format("csv")
             .option('header', 'true')
             .option('inferSchema', 'true')
             .load(offense_codes)
             # удаляем дубликаты записей.
             .dropDuplicates(['CODE']))
    
    # # удаляем дубликаты записей
    # os_df = oc_df_rows.dropDuplicates(['CODE'])
    
    # читаем файл offense_codes.csv
    crime_df = (spark.read.format('csv')
                .option('header', 'true')
                .option('inferSchema', 'true')
                .load(crime)
                # заполняем значения null.
                .fillna(value='No Name', subset='DISTRICT') 
                .fillna(value=0, subset=['Lat', 'Long'])                 
                # удаляем неиспользуемые колонки
                .drop(
                    'OFFENSE_CODE_GROUP',
                    'OFFENSE_DESCRIPTION',
                    'REPORTING_AREA',
                    'SHOOTING',
                    'DAY_OF_WEEK',
                    'YEAR',
                    'MONTH',
                    'HOUR',
                    'UCR_PART',
                    'STREET',
                    'Location'
                    ) 
                # приведение к дате
                .withColumn('occured', to_timestamp(col('OCCURRED_ON_DATE'), 'yyyy-MM-dd HH:mm:ss'))
                .drop('OCCURRED_ON_DATE')
                # обрезание даты до месяца
                .withColumn('year_month', date_trunc('month', col('occured')))
                .drop('occured').cache())
    
    # общее количество преступлений по каждому району
    crimes_total = (crime_df
                    .select('DISTRICT', 'OFFENSE_CODE')
                    .groupBy('DISTRICT')
                    .agg(count('OFFENSE_CODE').alias('crimes_total'))
                    )    
        
    # медиана числа преступлений в месяц по каждому району
    crimes_monthly = (crime_df
                      .select('DISTRICT', 'year_month', 'INCIDENT_NUMBER')
                      .groupBy('DISTRICT', 'year_month')
                      .agg(countDistinct('INCIDENT_NUMBER').alias('incidents'))
                      .orderBy(['DISTRICT', 'year_month'], ascending=[0, 1])
                      .groupBy('DISTRICT')
                      .agg(percentile_approx('incidents', 0.5, 10000).alias('crimes_monthly'))
                      .drop('year_month')
                      .withColumnRenamed('DISTRICT', 'DISTRICT_monthly')
                      )
    
    # Top-3 самых частых crime_type за всю историю наблюдений в этом районе.
    crime_df_name = crime_df.join(oc_df, crime_df.OFFENSE_CODE == oc_df.CODE, 'inner').drop('OFFENSE_CODE')
    
    district_code_df = (crime_df_name
                            .select('DISTRICT', 
                                    split(crime_df_name.NAME, '-')[0].alias('crime_type'), 
                                    'INCIDENT_NUMBER')                            
                            
                            .groupBy('DISTRICT', 'crime_type')
                            .agg(countDistinct('INCIDENT_NUMBER').alias('count_incidens'))
                            .orderBy(asc('DISTRICT'), desc('count_incidens'))
                            )
    
    
    win = Window.partitionBy('DISTRICT').orderBy(desc('count_incidens'))
    
    win_df = (district_code_df
                            .withColumn('inc', row_number().over(win))
                            .filter(condition=col('inc') < 4)
                            .withColumnRenamed('DISTRICT', 'DISTRICT_ct')
                            .select('DISTRICT_ct', 'crime_type', 'inc')
                            )
    
    # вычисляем lat и lng
    mean_lat = (crime_df
               .select('DISTRICT', 'Lat')
               .groupBy('DISTRICT')
               .agg(mean('Lat'))
               .withColumnRenamed('DISTRICT', 'DISTRICT_lat')
               .withColumnRenamed('avg(Lat)', 'lat')               
               )
    
    mean_long = (crime_df
               .select('DISTRICT', 'Long')
               .groupBy('DISTRICT')               
               .agg(mean('Long'))
               .withColumnRenamed('DISTRICT', 'DISTRICT_long')
               .withColumnRenamed('avg(Long)', 'lng')
               )
    
    # собираем результат.
    df = (crimes_total
          .join(crimes_monthly, crimes_total.DISTRICT == crimes_monthly.DISTRICT_monthly)  
          .drop('DISTRICT_monthly')        
          .join(win_df, crimes_total.DISTRICT == win_df.DISTRICT_ct)
          .filter(condition=col('inc') == 1)
          .drop('DISTRICT_ct').drop('inc')
          .withColumnRenamed('crime_type', 'crime_1')
          .join(win_df, crimes_total.DISTRICT == win_df.DISTRICT_ct)
          .filter(condition=col('inc') == 2)
          .drop('DISTRICT_ct').drop('inc')
          .withColumnRenamed('crime_type', 'crime_2')
          .join(win_df, crimes_total.DISTRICT == win_df.DISTRICT_ct)
          .filter(condition=col('inc') == 3)
          .drop('DISTRICT_ct').drop('inc')
          .withColumnRenamed('crime_type', 'crime_3')
          .withColumn('frequent_crime_types', concat(
                  trim(col('crime_1')), lit(', '), 
                  trim(col('crime_2')), lit(', '), 
                  trim(col('crime_3')))
                      )
          .drop('crime_1').drop('crime_2').drop('crime_3')          
          .join(mean_lat, crimes_total.DISTRICT == mean_lat.DISTRICT_lat)
          .drop('DISTRICT_lat')
          .join(mean_long, crimes_total.DISTRICT == mean_long.DISTRICT_long)          
          .drop('DISTRICT_long')
          .orderBy(asc('DISTRICT'))
          
    )
    
    # сохраняем результат.
    (df.write
     .format('parquet')
     .mode('overwrite')
     .option('compression', 'snappy')
     .save(path_to_folder))
    