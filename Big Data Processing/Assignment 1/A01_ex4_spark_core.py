# --------------------------------------------------------
#
# PYTHON PROGRAM DEFINITION
#
# The knowledge a computer has of Python can be specified in 3 levels:
# (1) Prelude knowledge --> The computer has it by default.
# (2) Borrowed knowledge --> The computer gets this knowledge from 3rd party libraries defined by others
#                            (but imported by us in this program).
# (3) Generated knowledge --> The computer gets this knowledge from the new functions defined by us in this program.
#
# When launching in a terminal the command:
# user:~$ python3 this_file.py
# our computer first processes this PYTHON PROGRAM DEFINITION section of the file.
# On it, our computer enhances its Python knowledge from levels (2) and (3) with the imports and new functions
# defined in the program. However, it still does not execute anything.
#
# --------------------------------------------------------

import pyspark
import time
from datetime import datetime
from datetime import timedelta


# ------------------------------------------
# FUNCTION process_line
# ------------------------------------------
def process_line(line):
    # 1. We create the output variable
    res = ()

    # 2. We get the parameter list from the line
    params_list = line.strip().split(",")

    #(00) Date => The date of the measurement. String <%Y-%m-%d %H:%M:%S> (e.g., "2013-01-01 13:00:02").
    #(01) Bus_Line => The bus line. Int (e.g., 120).
    #(02) Bus_Line_Pattern => The pattern of bus stops followed by the bus. String (e.g., "027B1001"). It can be empty (e.g., "").
    #(03) Congestion => On whether the bus is at a traffic jam (No -> 0 and Yes -> 1). Int (e.g., 0).
    #(04) Longitude => Longitude position of the bus. Float (e.g., -6.269634).
    #(05) Latitude = > Latitude position of the bus. Float (e.g., 53.360504).
    #(06) Delay => Delay of the bus in seconds (negative if ahead of schedule). Int (e.g., 90).
    #(07) Vehicle => An identifier for the bus vehicle. Int (e.g., 33304)
    #(08) Closer_Stop => An idenfifier for the closest bus stop given the current bus position. Int (e.g., 7486). It can be no bus stop, in which case it takes value -1 (e.g., -1).
    #(09) At_Stop => On whether the bus is currently at the bus stop (No -> 0 and Yes -> 1). Int (e.g., 0).

    # 3. If the list contains the right amount of parameters
    if (len(params_list) == 10):
        # 3.1. We set the right type for the parameters
        params_list[1] = int(params_list[1])
        params_list[3] = int(params_list[3])
        params_list[4] = float(params_list[4])
        params_list[5] = float(params_list[5])
        params_list[6] = int(params_list[6])
        params_list[7] = int(params_list[7])
        params_list[8] = int(params_list[8])
        params_list[9] = int(params_list[9])

        # 3.2. We assign res
        res = tuple(params_list)

    # 4. We return res
    return res

def calculatetime(curent_t,aftertime,listof_date):
        if ((datetime.strptime(curent_t, '%Y-%m-%d %H:%M:%S')) < (datetime.strptime(listof_date, '%Y-%m-%d %H:%M:%S')) <  (datetime.strptime(aftertime, '%Y-%m-%d %H:%M:%S'))):
             return listof_date

def getdaysbusline(daysbulinelist):
    days, busline,stop = daysbulinelist
    return (days, [(busline,stop)])
# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(sc, my_dataset_dir, current_time, current_stop, seconds_horizon):
    # 1. Operation C1: 'textFile' to load the dataset into an RDD
    inputRDD = sc.textFile(my_dataset_dir)
    inputRDD1 = inputRDD.map(lambda line: line.split(","))
    dattimef = inputRDD1.map(lambda a: a[0])

    #calculate the endtime from current time and second_horizon
    endtime = (datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=seconds_horizon)).strftime("%Y-%m-%d %H:%M:%S")

    #select the data between the current time and endtime
    inputRDD2 =  inputRDD1.filter(lambda line: calculatetime(current_time,endtime, line[0]))

    #filter the data of current stop and atstop
    inputRDD2 = inputRDD2.filter(lambda x: x[8] == str(current_stop) and x[9] == "1")

    #select the first vehicle id
    vehicalID = inputRDD2.first()[7]

    #select the data on vehicle id and current time and end time
    inputRDD2 = inputRDD1.filter(lambda line : line[7] == vehicalID and line[9] == "1" and calculatetime(current_time,endtime, line[0]))

    #select the data only vehicle id,date and stop
    inputRDD4 = inputRDD2.map(lambda p : (int(p[7]),p[0],int(p[8])))

    #combine the date and stop together
    solutionRDD4 = inputRDD4.map(getdaysbusline).sortByKey()

    # filter the data on key value pair
    solutionRDD3 = solutionRDD4.reduceByKey(lambda p, q: p + q)

    #collect the data in list
    resVAL = solutionRDD3.collect()

    for item in resVAL:
        print(item)

# --------------------------------------------------------
#
# PYTHON PROGRAM EXECUTION
#
# Once our computer has finished processing the PYTHON PROGRAM DEFINITION section its knowledge is set.
# Now its time to apply this knowledge.
#
# When launching in a terminal the command:
# user:~$ python3 this_file.py
# our computer finally processes this PYTHON PROGRAM EXECUTION section, which:
# (i) Specifies the function F to be executed.
# (ii) Define any input parameter such this function F has to be called with.
#
# --------------------------------------------------------
if __name__ == '__main__':
    # 1. We use as many input arguments as needed
    current_time = "2013-01-10 08:59:59"
    current_stop = 1935
    seconds_horizon = 1800

    # 2. Local or Databricks
    local_False_databricks_True = False

    # 3. We set the path to my_dataset and my_result
    my_local_path = "D:/Study/BDP/A01_dataset/"
    my_databricks_path = "/"
    my_dataset_dir = "my_dataset_complete/"

    if local_False_databricks_True == False:
        my_dataset_dir = my_local_path + my_dataset_dir
    else:
        my_dataset_dir = my_databricks_path + my_dataset_dir

    # 4. We configure the Spark Context
    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel('WARN')
    print("\n\n\n")

    # 5. We call to our main function
    my_main(sc, my_dataset_dir, current_time, current_stop, seconds_horizon)
