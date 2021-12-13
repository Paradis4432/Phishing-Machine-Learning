import pandas as pd

def replace(df, column, find, value):
    df[column] = df[column].replace(find, value)
    return df
    
def replaceBool(df, columnList):
    for i in columnList:
        df[i] = df[i].astype(int)
    return df

def getCleanData():
    """
    This function will return a cleaned dataframe
    Replacing Encoding variables with categorical variables
    And True for 1 / False for 0
    """
    enron = pd.read_csv("./dataScience/data/features-enron.csv")
    phi = pd.read_csv("./dataScience/data/features-phishing.csv")

    frames = [enron, phi]
    full = pd.concat(frames).drop(columns=['Unnamed: 0']).sample(frac=1)

    # reemplazar valores en Encoding a valores categoricos
    full = replace(full, "Encoding", ["7bit", "7bit "], 1)
    full = replace(full, "Encoding", "none", 2)
    full = replace(full, "Encoding", "quoted-printable", 3)
    full = replace(full, "Encoding", "8bit", 4)
    full = replace(full, "Encoding", "8bit\\r\\n", 5)
    full = replace(full, "Encoding", "base64", 6)
    full = replace(full, "Encoding", "7bit\n\tboundary=\"--vhoabg67774\"", 7)

    # reemplazar valores True/False a 1 y 0 en 
    # @ in URLs Flash content	HTML content	Html Form	Html iFrame	IPs in URLs
    full = replaceBool(full, ["@ in URLs", "Flash content", "HTML content","Html Form",
    "Html iFrame", "IPs in URLs", "Phishy"])
    return full 