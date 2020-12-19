import pandas as pd
pd.set_option("max_columns", None)
pd.set_option("max_rows", None)


csv_file = '../data/x_train.csv'



def main():
    data = pd.read_csv(csv_file,dtype='float32')
    dates = data['date']
    features = data.iloc[:,8:]
    print(features.describe())
    #print(features['feature_7'])
    means = features.mean()
    #print(features.fillna(means)['feature_8'])
    print(dates.describe())

    print('num of pos resp: ',len(data[data['resp']>0]))
    print('num of neg resp: ',len(data[data['resp'] < 0]))


if __name__ == "__main__":
    main()