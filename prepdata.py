def prepare_data(fullpathInput, fullpathOutput):
    
    import time
    start_time = time.time()
    
    import pandas as pd
    DF = pd.read_csv(fullpathInput, sep='|', low_memory = False)
    
    print("Start parsing the data...")
    
    import numpy as np

    #print(DF.columns.values)
    featureNames = DF.columns.values
    numFeatures = len(DF.columns.values)

    requiredFeatures = ['Partition','ReqNodes','ReqCPUS','NNodes','Timelimit','Submit','Start','End','Eligible','QueueTime']

    out = set(featureNames).intersection(requiredFeatures)
    if (len(out)!=len(requiredFeatures)):
        missing = [feature for feature in requiredFeatures if feature not in featureNames]
        raise ValueError("The following Features: %s are missing (or mispelled) in the provided dataset" % missing)

    workingDataset = DF.loc[:,requiredFeatures]

    # Convert q times to integers (in seconds)
    print("------------------------------------Parse the feature 'QueueTime'-------------------------------------")
    QTseries = workingDataset.loc[:,"QueueTime"]
    # A better strategy to deal with non formatted time is needed to make the algorithm more robust
    ld = QTseries.str.contains("day")
    if any(ld): 
        print("special convertion from days is needed \ndetecting number of days...")
        QTdays = QTseries.loc[ld]
        days = []
        for day in QTdays:
            dayFormat = day.split(" ")
            days.append(int(dayFormat[0])*24)
        print("No. %i entries detected with unkown time format of type 'days' for qTimes."%ld.values.sum())
        QTseries.loc[ld] = days
    
    QTseries.loc[~ld] = QTseries.loc[~ld].str.split(':').apply(lambda x: int(x[0]) * 60 * 60 + int(x[1]) * 60 + int(x[2]))
    print("------------------------------------------------------------------------------------------------------")


    # Convert the time limits to integers (in seconds)
    print("------------------------------------Parse the feature 'Timelimit'-------------------------------------")
    TLseries = workingDataset.loc[:,"Timelimit"]
    ltl_u = TLseries.str.contains("UNLIMITED")
    ltl_p = TLseries.str.contains("Partition_Limit")
    ltl_d = TLseries.str.contains("-")
    ltd_up = (ltl_u | ltl_p)
    ltl_udp = (ltl_u | ltl_d | ltl_p)

    if any(ltl_p): 
        print("No. %i entries of 'Partition_Limit' type detected. Partition_Limit will be converted into a large integer (3 years)"%ltl_p.values.sum())
        TLseries.loc[ltl_p] = int(10**8)
    
    if any(ltl_u): 
        print("No. %i entries of 'UNLIMITED' type detected. 'UNLIMITED' time will be converted into a large integer (3 years)"%ltl_u.values.sum())
        TLseries.loc[ltl_u] = int(10**8)
      
    if any(ltl_d):
        print("No. %i entries expressed in days. These values will be converted in seconds."%ltl_d.values.sum())
        TLseries.loc[ltl_d] = TLseries.loc[ltl_d].str.replace('-',':')
      
    y=np.array(range(0,int(ltl_udp.size)))
    Zs = pd.Series(["00:"]*y[~ltl_udp.values].size, index=y[~ltl_udp.values])
    TLseries.loc[~ltl_udp] = Zs.str.cat(TLseries.loc[~ltl_udp])
    
    TLseries.loc[~ltd_up] = TLseries.loc[~ltd_up].str.split(':').apply(lambda x: int(x[0]) * 24 * 60 * 60 + int(x[1]) * 60 * 60 + int(x[2]) * 60 + int(x[3]))
    
    workingDataset.to_csv(fullpathOutput,sep=",")
    
    print("-------------------------------------------------------------------------------------------------------")
    
    print("Total time needed to parse %i entries:" % workingDataset.size)
    print("--- %s seconds ---" % (time.time() - start_time))