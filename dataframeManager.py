import pandas as pd
import os

class DataFrameManager:

    dfDict = {}
    def __init__(self):
        pass

    def create_new_df(self, dataframename: str):
        newDf = pd.DataFrame(columns = ['aspectRatio', 'area', 'perimeter',
                                        'rectangularity', 'circularity', 'equiDiameter',
                                        'angle', 'classification'])
        tempDict = {dataframename: newDf}
        self.dfDict.update(tempDict)
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
        print("Appending to dataframe: " + dataframename + '\n Data to append: ' + str(data))
        rowDf = pd.DataFrame(data)
        row = {'aspectRatio': data[0], 'area': data[1], 'perimeter': data[2],
                                                     'rectangularity': data[3], 'circularity': data[4],
                                                     'equiDiameter': data[5],
                                                     'angle': data[6], 'classification': data[7]}
        dfAppend = self.get_df(dataframename)
        # dfAppend.reset_index(drop=True, inplace=True)
        # rowDf.reset_index(drop=True, inplace=True)
        dfAppend = dfAppend.append(row, ignore_index=True)
        self.del_df(dataframename)
        tempDict = {dataframename: dfAppend}
        self.dfDict.update(tempDict)
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

    def print_df(self, dataframename: str):
        print("DATAFRAME IN PANDAS BEFORE EXPORT")
        df = self.get_df(dataframename=dataframename)
        print(df.to_string())

