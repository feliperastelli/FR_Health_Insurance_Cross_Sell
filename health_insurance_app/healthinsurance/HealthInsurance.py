import pickle
import numpy  as np
import pandas as pd

class HealthInsurance(object):
    
    def __init__( self ):
        self.home_path = ''
        self.annual_premium_scaler =            pickle.load( open( self.home_path + 'features/annual_premium_scaler.pkl', 'rb' ) )
        self.age_scaler =                       pickle.load( open( self.home_path + 'features/age_scaler.pkl', 'rb' ) ) 
        self.vintage_scaler =                   pickle.load( open( self.home_path + 'features/vintage_scaler.pkl', 'rb' ) ) 
        self.target_encode_gender_scaler =      pickle.load( open( self.home_path + 'features/target_encode_gender_scaler.pkl', 'rb' ) )
        self.target_encode_region_code_scaler = pickle.load( open( self.home_path + 'features/target_encode_region_code_scaler.pkl', 'rb' ) )
        self.fe_policy_sales_channel_scaler =   pickle.load( open( self.home_path + 'features/fe_policy_sales_channel_scaler.pkl','rb' ) )
        
    def data_cleaning( self, df1 ):
        
        df1['policy_sales_channel'] = df1['policy_sales_channel'].astype(int)
        df1['region_code'] = df1['region_code'].astype(int)

        return df1 
    
    def feature_engineering( self, df2 ):
        
        # Change feature vehicle age 
        df2['vehicle_age'] = df2['vehicle_age'].apply( lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_year' if x == '1-2 Year' else 'below_1_year')

        # Change feature vehicle damage
        df2['vehicle_damage'] = df2['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0)
        
        return df2
    
    def data_preparation( self, df3 ):
        
        # anual premium - StandardScaler
        df3['annual_premium'] = self.annual_premium_scaler.transform( df3[['annual_premium']].values )

        # Age - MinMaxScaler
        df3['age'] = self.age_scaler.transform( df3[['age']].values )

        # Vintage - MinMaxScaler
        df3['vintage'] = self.vintage_scaler.transform( df3[['vintage']].values )

        # gender - Target Encoding
        df3.loc[:, 'gender'] = df3['gender'].map( self.target_encode_gender_scaler )

        # region_code - Target Encoding
        df3.loc[:, 'region_code'] = df3['region_code'].map( self.target_encode_region_code_scaler )

        # vehicle_age - One Hot Encoding
        df3 = pd.get_dummies( df3, prefix='vehicle_age', columns=['vehicle_age'] )

        # policy_sales_channel - Frequency Encoding
        df3.loc[:, 'policy_sales_channel'] = df3['policy_sales_channel'].map( self.fe_policy_sales_channel_scaler )
        
        # Feature Selection
        cols_selected = ['annual_premium', 'vintage', 'age', 'region_code', 'vehicle_damage', 'previously_insured',
                         'policy_sales_channel']
        
        return df3[ cols_selected ]
    
    
    def get_prediction( self, model, original_data, test_data ):
        
        # model prediction
        pred = model.predict_proba( test_data )
        
        # join prediction into original data
        original_data['score'] = pred[:,1].tolist()
        
        return original_data.to_json( orient='records', date_format='iso' )
        
