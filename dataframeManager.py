import pandas as pd
import os

class DataFrameManager:

    dfDict = {}
    def __init__(self):
        pass

    def create_new_df(self, dataframename: str):
        newDf = pd.DataFrame(columns = ['aspectRatio', 'area', 'perimeter', 'thresholdImg',
                                        'rectangularity', 'circularity', 'equiDiameter',
                                        'angle', 'classification'])
        self.dfDict.update(dataframename, newDf)
        return

    def has_df(self, dataframename: str) -> bool:
        if dataframename in self.dfDict:
            return True
        else:
            return False

    def del_df(self, dataframename: str):
        if self.has_df(dataframename):
            self.dfDict.pop(dataframename, None)
        else:
            print("could not del_df. no dataframe called " + dataframename + 'exists')

    def get_df(self, dataframename: str) -> pd.DataFrame:
        if self.has_df(dataframename):
            return self.dfDict[dataframename]
        else:
            print("could not get_df. no dataframe called " + dataframename + 'exists')

    def append_to_df(self, dataframename: str, data: list):
        rowDf = pd.DataFrame([data[0], data[1], data[2], data[3], data[4],
                              data[5], data[6], data[6], data[7]])
        dfAppend = self.get_df(dataframename)
        dfAppend = pd.concat([rowDf, dfAppend], ignore_index=True)
        self.del_df(dataframename)
        self.dfDict.update(dataframename, dfAppend)
        return

    def export_df(self, dataframename: str) -> bool:
        if self.has_df(dataframename):
            path = os.getcwd() + '\\dataframes\\' + dataframename + '.csv'
            df = self.get_df(dataframename)
            try:
                df.to_csv(path)
                return True
            except:
                print("Could not export " + dataframename + " to a .csv file")
                return False




