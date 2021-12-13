
import dataManager as dm
import modelManager as mm

# gets a clean df from the dataManager
# df.Encoding is a categorical variable
# replaced bool with 0 and 1 in:
# "@ in URLs", "Flash content", "HTML content","Html Form",
# "Html iFrame", "IPs in URLs", "Phishy"
cleanDF = dm.getCleanData()

# print(cleanDF.sample(15))

# creates a trained model
# model is trained on the cleanDF
# model is saved as a pickle file
"""
@param: dataFrame
@param: targetColumn
@param: save model
@param: return dictioanry of model
@return: dictionary of models
"""
def generateAllModels(returnDict):
    NBmodel = mm.makeNBmodel(cleanDF, "Phishy", True, True)
    LGmodel = mm.makeLGmodel(cleanDF, "Phishy", True, True)
    KNNmodel = mm.makeKNNmodel(cleanDF, "Phishy", True, True)
    if returnDict:
        return {
            "NB": NBmodel,
            "LG": LGmodel,
            "KNN": KNNmodel
        }
