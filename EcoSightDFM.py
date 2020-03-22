import pandas as pd


class DataFrameManager:

    dfDict = {}

    def __init__(self):
        pass

    def createnewdataframe(self, dataframename: str) -> bool:
        if dataframename in self.dfDict:
            print("Dataframe" + dataframename + "already exists in the dataframe mangager \n")
            return False
        else:
            newDataFrame = pd.DataFrame(columns={"Img.Datastd", "Img.DataGS","Img.DataBlurred","Img.DataMorph",
                                                  "Aspect Ration","Rectangularity","Circularity","Classification"})
            self.dfDict[dataframename] = newDataFrame
            if dataframename in self.dfDict:
                return True
            else:
                print("Not able to create the dataframe " + dataframename)
                return False

    def adddf(self, dataframename: str,  dataframe: pd.DataFrame):
        if dataframename in self.dfDict:
            print("Dataframe"+ dataframename + "already exists in the dataframe mangager \n")
            return
        else:
            self.dfDict[dataframename] = dataframe

    def hasdf(self, dataframename: str) -> bool:
        if dataframename in self.dfDict:
            return True
        else:
            return False

    def getdf(self, dataframename: str) -> pd.DataFrame:
        if dataframename in self.dfDict:
            return self.dfDict[dataframename]
        else:
            print("Could not find " + dataframename + "in the dataframe manager \n")
            pass

    def deletedf(self, dataframename: str):
        if dataframename in self.dfDict:
            del self.dfDict[dataframename]
            return
        else:
            print("Could not find " + dataframename + "in the dataframe manager \n")
            return

    def appendtodataframe(self, dataframename: str, values: list) -> pd.DataFrame:
        if dataframename in self.dfDict:
            dataframe = self.getdf(dataframename)
            newdf = dataframe.append(values, ignore_index=True)
            self.deletedf(dataframename=dataframename)
            self.adddf(dataframename, newdf)
            return newdf
        else:
            print("Could not find " + dataframename + "in the dataframe manager \n")
            pass

    def printdataframes(self):
        for df in self.dfDict:
            print(df)
