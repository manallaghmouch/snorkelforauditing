import pandas as pd
import numpy as np

#from pm4py.objects.log.importer.xes import factory as xes_import_factory
#train_log = xes_import_factory.apply()

df_train0 = pd.read_csv('/home/manal/Downloads/Audit_1000.csv', sep=';')
df_train1 = pd.read_csv('/home/manal/Downloads/Real_1000.csv', sep=';')
df_valid0 = pd.read_csv('/home/manal/Downloads/Audit_500.csv', sep=';')
df_valid1 = pd.read_csv('/home/manal/Downloads/Real_500.csv', sep=';')

# EVENT LOG TO TRACE LOG
# Preprocess training data: one case per row
df_train_trace = df_train0.groupby("Case ID")
traces_train = []

for n in range(1,10):
    trace1 = df_train_trace.get_group("Case No. 000" + str(n))
    list_trace1 = trace1['Activity'].tolist()
    traces_train.append(list_trace1)

for n in range(10,100):
    trace2 = df_train_trace.get_group("Case No. 00" + str(n))
    list_trace2 = trace2['Activity'].tolist()
    traces_train.append(list_trace2)

for n in range(100,1000):
    trace3 = df_train_trace.get_group("Case No. 0" + str(n))
    list_trace3 = trace3['Activity'].tolist()
    traces_train.append(list_trace3)

for n in range(1000,1001):
    trace4 = df_train_trace.get_group("Case No. " + str(n))
    list_trace4 = trace4['Activity'].tolist()
    traces_train.append(list_trace3)

df_train_0 = pd.DataFrame({'case_id':np.arange(len(traces_train))+1,'trace':list(traces_train),
                         'tracestring':''})

df_train_trace = df_train1.groupby("Case ID")
traces_train = []

for n in range(1,10):
    trace1 = df_train_trace.get_group("Case No. 000" + str(n))
    list_trace1 = trace1['Activity'].tolist()
    traces_train.append(list_trace1)

for n in range(10,100):
    trace2 = df_train_trace.get_group("Case No. 00" + str(n))
    list_trace2 = trace2['Activity'].tolist()
    traces_train.append(list_trace2)

for n in range(100,1000):
    trace3 = df_train_trace.get_group("Case No. 0" + str(n))
    list_trace3 = trace3['Activity'].tolist()
    traces_train.append(list_trace3)

for n in range(1000,1001):
    trace4 = df_train_trace.get_group("Case No. " + str(n))
    list_trace4 = trace4['Activity'].tolist()
    traces_train.append(list_trace3)

df_train_1 = pd.DataFrame({'case_id':np.arange(len(traces_train))+1,'trace':list(traces_train),
                         'tracestring':''})

df_train_1['case_id'] = df_train_1['case_id'] + 1000
df_train = [df_train_0, df_train_1]
df_train = pd.concat(df_train)

#df_train['tracestring'] = df_train['tracestring'].astype('str')
df_train['tracestring'] = [','.join(map(str,l)) for l in df_train['trace']]

# Preprocess validation set
df_valid_trace = df_valid0.groupby("Case ID")
traces_valid0 = []

for n in range(1,10):
    trace1 = df_valid_trace.get_group("Case No. 00" + str(n))
    list_trace1 = trace1['Activity'].tolist()
    traces_valid0.append(list_trace1)

for n in range(10,100):
    trace2 = df_valid_trace.get_group("Case No. 0" + str(n))
    list_trace2 = trace2['Activity'].tolist()
    traces_valid0.append(list_trace2)

for n in range(100,501):
    trace2 = df_valid_trace.get_group("Case No. " + str(n))
    list_trace2 = trace2['Activity'].tolist()
    traces_valid0.append(list_trace2)

deviations_df0 = pd.DataFrame({'case id': df_valid0['Case ID'], 'deviation': df_valid0['Deviation']})
deviations_df0 = deviations_df0.drop_duplicates()
#deviations_df0 = deviations_df0[:-1] ##############


df_valid_trace = df_valid1.groupby("Case ID")
traces_valid1 = []
# deze for-loop heb ik anders opgesteld dan de vorige, omdat de gebruikte dataset afgekapt is (van 1000 naar 500)
for n in range(1,10):
    trace1 = df_valid_trace.get_group("Case No. 000" + str(n))
    list_trace1 = trace1['Activity'].tolist()
    traces_valid1.append(list_trace1)

for n in range(10,100):
    trace2 = df_valid_trace.get_group("Case No. 00" + str(n))
    list_trace2 = trace2['Activity'].tolist()
    traces_valid1.append(list_trace2)

for n in range(100,501):
    trace2 = df_valid_trace.get_group("Case No. 0" + str(n))
    list_trace2 = trace2['Activity'].tolist()
    traces_valid1.append(list_trace2)

deviations_df1 = pd.DataFrame({'case id': df_valid1['Case ID'], 'deviation': df_valid1['Deviation']})
deviations_df1 = deviations_df1.drop_duplicates()
deviations_df1 = deviations_df1[:-1]

df_valid_0 = pd.DataFrame({'case_id':np.arange(len(traces_valid0))+1,'trace':list(traces_valid0),
                        'deviation':deviations_df0['deviation'],'tracestring':''})
df_valid_1 = pd.DataFrame({'case_id':np.arange(len(traces_valid1))+1,'trace':list(traces_valid1),
                        'deviation':deviations_df1['deviation'],'tracestring':''})

df_valid_1['case_id'] = df_valid_1['case_id'] + 500
df_valid = [df_valid_0, df_valid_1]
df_valid = pd.concat(df_valid)
#df_valid['deviation'].replace({"FP": 0, "TP": 1}, inplace=True)

Y_valid = df_valid["deviation"]


df_valid['tracestring'] = [','.join(map(str,l)) for l in df_valid['trace']]
