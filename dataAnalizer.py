import pandas as pd
from sklearn.model_selection import GroupKFold

pd.set_option("max_columns", None)
pd.set_option("max_rows", None)


csv_file = '../data/train.csv'



def main():
    data = pd.read_csv(csv_file,dtype='float32')

    dates = data['date']
    #features = data.iloc[:,data.columns.str.contains('feature')]

    dates = dates.unique()
    print(dates, len(dates))
    #print(features.describe())
    #print(features['feature_7'])
    #means = features.mean()
    #print(features.fillna(means)['feature_8'])
    #print(dates.describe())

    #print('num of pos resp: ',len(data[data['resp']>0]))
    #print('num of neg resp: ',len(data[data['resp'] < 0]))

    print('*'*30)

    # naCount = features.isna().sum()
    # naPct = naCount/len(features)
    # #print(naPct)
    # nllColumns = naPct[naPct > 0.1]
    # print(nllColumns)


    # dates = dates.unique()
    # d = data[ data['date'] == 0 ]
    # dm = d.mean()
    # date_means = pd.DataFrame(data=None,columns=data.columns)
    # date_means[0] = dm
    # print(date_means)

    # print(d - date_means[0])

if __name__ == "__main__":
    main()