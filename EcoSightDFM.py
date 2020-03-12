import pandas

class dataFrameManager:

    dfDict = {}
    def __init__(self):
        return

    def createnewdataframe(dataframename: str) -> bool:
        if dataframename in dfDict:
            print("Dataframe"+ dataframename + "already exists in the dataframe mangager \n")
            return False
        else
            newDataFrame = pd.DataFrame(columns= {"Img.Datastd", "Img.DataGS","Img.DataBlurred","Img.DataMorph",
                                         "Aspect Ration","Rectangularity","Circularity","Classification"})
            dfDict[dataframename] = newDataFrame
            if dataframename in dfdict:
                return True
            else
                print("Not able to create the dataframe " + dataframename)
                return False
    def adddf(self,dataframename: str,  dataframe: pd.DataFrame):
        if dataframename in dfDict:
            print("Dataframe"+ dataframename + "already exists in the dataframe mangager \n")
            return
        else:
            dfDict[dataframename] = dataframe
    def getdf(dataframename: str) -> pd.DataFrame:
        if dataframename in dfDict:
            return dfDict[dataframename]
        else
            print("Could not find " + dataframename + "in the dataframe manager \n")
            return
    def deletedf(dataframename: str):
        if dataframename in dfDict:
            dfDict.delete(dataframename)
            return
        else
            print("Could not find " + dataframename + "in the dataframe manager \n")
            return

    def appendtodataframe(dataframename: str, values: list) -> pd.DataFrame:
        if dataframename in dfDict:
            dataframe = self.getdf(dataframename)
            newDf = dataframe.append({values[0, values[1], values[2], values[3],
                                     values[4], values[5], values[6], values[7]}, ignore_index=True)
            self.deletedf(dataframename)
            self.adddf(newDf)
            return newDf
        else:
            print("Could not find " + dataframename + "in the dataframe manager \n")

    def printdataframes(self):
        for df in dfDict:
            print(df)
