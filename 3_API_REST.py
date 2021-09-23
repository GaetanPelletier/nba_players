from flask import Flask
from flask_restplus import Api, Resource

import pandas as pd
import numpy as np
import dill as pickle

'''
Attention, pour fonctionner, il est necessaire d'installer les packages suivants:
flask==1.1.2
flask_restplus==0.13.0
scikit-learn==0.22.2.post1
Werkzeug==0.16.1
'''

app=Flask(__name__)

api = Api(app=app, version='0.1', title='Prediction recrutement joueur NBA - Gaetan Pelletier', description='', validate=True)

# model
with open("model/model_logReg.pk", "rb") as f:
    model = pickle.load(f)

# seuil
    threshold = 0.51

@api.route('/nba_player/<int:gp>/<float:pts>/<float:fg_percentage>/<float:threep_percentage>/<float:ft_percentage>/<float:oreb>/<float:dreb>/<float:ast>/<float:stl>/<float:blk>/<float:tov>/')
class prediction_recrutement(Resource):
    def get(self, gp, pts, fg_percentage, threep_percentage, ft_percentage, oreb, dreb, ast, stl, blk, tov):
        """
        Retourne classe positive ou negative suivant les caractéristiques du joueur
        """

        # recupere data
        df = pd.DataFrame({
            "gp": [gp],
            "pts": [pts],
            "fg_p": [fg_percentage],
            "threep_p": [threep_percentage],
            "ft_p": [ft_percentage],
            "oreb": [oreb],
            "dreb": [dreb],
            "ast": [ast],
            "stl": [stl],
            "blk": [blk],
            "tov": [tov],
        })

        # A integrer dans pipeline du modele...
        # 7 col parmi 11 -> np.log(x+1)
        df.pts = np.log(df.pts + 1)
        df.oreb = np.log(df.oreb + 1)
        df.dreb = np.log(df.dreb + 1)
        df.ast = np.log(df.ast + 1)
        df.stl = np.log(df.stl + 1)
        df.blk = np.log(df.blk + 1)
        df.tov = np.log(df.tov + 1)        

        # pred
        y_pred = model.predict_proba(df)        

        # si pred >= seuil -> 1 -> "Joueur a recruter !"
        # sinon 0 -> "Joueur a ne pas recruter"        
        if y_pred[0][1] >= threshold:
            return {"Result = 1 -> Joueur a recruter !"}
        else:
            return {"Result = 0 -> Joueur a ne pas recruter..."}
    
if __name__=="__main__":
    app.run(port=5000, host='localhost', debug=True)