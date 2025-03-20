#-------------------------------------------------------------------------------
#                              ROSSMANN PROJECT
# File: handler.py
#
# Goal: this ML-API receives/solves a request to perform a forecast for one store
#       Implements ML prediction model, using Rossmann class (Rossmann.py file)
#
#                                                        First date.: 2023.12.20
#                                                        Last update: 2024.01.16
#-------------------------------------------------------------------------------

import os
import pickle
import pandas as pd
from flask             import Flask, request, Response
from rossmann.Rossmann import Rossmann

#............... Load model
model = pickle.load( open( 'model/model_rossmann.pkl', 'rb' ) )

#............... Init API
# REF: https://flask.palletsprojects.com/en/2.3.x/genindex/

app = Flask( __name__ )

#............... Create Endpoint Route
# REF: https://flask.palletsprojects.com/en/2.3.x/api/#flask.Blueprint.route
# rossmann_predict function: that's the handler, activated when the API receives a request

@app.route( '/rossmann/predict', methods=['POST'] )
def rossmann_predict():
    test_json = request.get_json()

    if test_json: # is there any data in the received request?
        if isinstance( test_json, dict ):
            # Convert 1-line Json to Dataframe
            test_raw = pd.DataFrame( test_json, index=[0] )
        else:
            # Convert N-lines Json to Dataframe
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
        
        # Instantiate Rossmann class
        pipeline = Rossmann()

        # data cleaning
        df1 = pipeline.data_cleaning( test_raw )
        # feature engineering
        df2 = pipeline.feature_engineering( df1 )
        # data preparation
        df3 = pipeline.data_preparation( df2 )
        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df3 )

        return df_response

    else:
        # No data in the received request:
        return Response( '{}', status=200, mimetype='application/json' )


#............... Run Flask API
if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000 )
    app.run( '0.0.0.0', port=port )
