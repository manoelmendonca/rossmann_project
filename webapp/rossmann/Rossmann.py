#-------------------------------------------------------------------------------
#                              ROSSMANN PROJECT
# File: Rossmann.py
#
# Goal: this Rossmann Prediction Class is used in 'handler.py' file, being...
#       ...called by the MachineLearning-API.
#       The class encapsulates the commands to perform a ML forecasting
#       It performs all the data clearing, encodings & transformation needed.
#
#                                                        First date.: 2023.12
#                                                        Last update: 2025.03.18
#-------------------------------------------------------------------------------

import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class DataConversion:

    def __init__(self):
        # All scalers (MinMax, Robust, etc) here.
        self.promo_time_week_scaler = None
        self.competition_distance_scaler = None
        self.competition_time_month_scaler = None
        self.store_type_scaler = None
        self.state_holiday_scaler = None
        self.year_scaler = None

    def get_scalers(self):
        """
        Returns the fitted scalers and encoders.
        """
        return {
            'promo_time_week': self.promo_time_week_scaler,
            'competition_distance': self.competition_distance_scaler,
            'competition_time_month': self.competition_time_month_scaler,
            'store_type': self.store_type_scaler,
            'state_holiday': self.state_holiday_scaler,
            'year': self.year_scaler
        }

    def set_scalers(self, inDict):
        """
        Receives a dictionary of scalers/encoders and load.
        (to be used in Rossmann class)
        """
        self.promo_time_week_scaler        = inDict['promo_time_week']
        self.competition_distance_scaler   = inDict['competition_distance']
        self.competition_time_month_scaler = inDict['competition_time_month']
        self.store_type_scaler    = inDict['store_type']
        self.state_holiday_scaler = inDict['state_holiday']
        self.year_scaler          = inDict['year']

    def feature_engineering( self, in_df ):
        # Create/adjust features based solely on the current register, no need for fit-transform
        self.prepare_time( in_df )
        self.prepare_promo2( in_df )
        self.prepare_competition( in_df )
        self.prepare_assortment_string( in_df )
        self.prepare_state_holiday( in_df )
        #self.prepare_sales_per_customer( in_df )

    def data_preparation_fit( self, in_df ):
        # 
        #self.prepare_response_variable( in_df )
        self.fit_promo_time_week( in_df )
        self.fit_competition_distance( in_df )
        self.fit_competition_time_month( in_df )
        self.fit_state_holiday( in_df )
        self.fit_store_type( in_df )
        self.prepare_assortment_num( in_df )
        self.fit_year( in_df )
        self.prepare_time_features( in_df )

    def data_preparation_transform( self, in_df ):
        #
        #self.prepare_response_variable( in_df )
        self.transform_promo_time_week( in_df )
        self.transform_competition_distance( in_df )
        self.transform_competition_time_month( in_df )
        self.transform_state_holiday( in_df )
        self.transform_store_type( in_df )
        self.prepare_assortment_num( in_df )
        self.transform_year( in_df )
        self.prepare_time_features( in_df )


    #.......... FEATURE ENGINEERING AUXILIARY METHODS

    def prepare_promo2(self, df3):
        # Create attribute IS_PROMO2:
        # from "promo2", "promo2_since_year/week" and "promo_interval"
        # forced to ZERO if date < (promo2_since_week & promo2_since_year)
        # PS: this is necessary because, originally, 'promo2' doesn't vary through time.
        # Example: STORE-28 started its 'promo2' only in 2015, 6th week. Before this week, promo2 should be ZERO.
        df3['is_promo2'] = df3.apply(
            lambda x: 1 if (
                (x['promo2'] == 1) and
                ((x['date'].year > x['promo2_since_year']) or
                ((x['date'].year == x['promo2_since_year']) and (x['date'].week >= x['promo2_since_week'])))
                and (x['date'].strftime('%b') in (x['promo_interval'] or ''))
            ) else 0,
            axis=1
        )

        # Attribute PROMO2_SINCE
        # PS: using features: "promo2", "since_week" and "since_year"
        promo2_str = df3['promo2_since_year'].astype(str) + '-' + df3['promo2_since_week'].astype(str) + '-1'
        df3['promo2_since'] = pd.to_datetime(promo2_str, format='%Y-%W-%w') - pd.Timedelta(days=7)

        # Attribute PROMO2_TIME_WEEK
        df3['promo2_time_week'] = np.maximum(((df3['date'] - df3['promo2_since']).dt.days // 7)
                                             .fillna(0).astype(int), 0)

    def prepare_competition(self, df3):

        date_year = df3['date'].dt.year
        date_month = df3['date'].dt.month

        # Create HAS_COMPETITION: =ZERO in the days before CompetitionOpenSince[month/year].
        #                         =ONE  for registers after that date.
        df3['has_competition'] = (
            (date_year > df3['competition_open_since_year']) |
            ((date_year == df3['competition_open_since_year']) & (date_month >= df3['competition_open_since_month']))
        ).astype(int)

        # Adjust COMPETITION_DISTANCE
        df3.loc[df3['has_competition'] == 0, 'competition_distance'] = 200000.0

        # Create COMPETITION_SINCE
        # PS: info separated in MONTH & YEAR. Join them and calculate date difference to current date
        df3['competition_since'] = pd.to_datetime(
            df3['competition_open_since_year'].astype(str) + '-' + 
            df3['competition_open_since_month'].astype(str) + '-01', 
            format='%Y-%m-%d'
        )
        # Create COMPETITION_TIME_MONTH:
        df3['competition_time_month'] = np.maximum(((df3['date'] - df3['competition_since']).dt.days // 30)
                                                   .fillna(0).astype(int), 0)

    def prepare_time(self, df3):
        # year
        df3['year'] = df3['date'].dt.year.astype( np.int64 )
        # month
        df3['month'] = df3['date'].dt.month.astype( np.int64 )
        # day
        df3['day'] = df3['date'].dt.day.astype( np.int64 )
        # semester
        df3['semester'] = (df3['month'] >= 7).astype(np.int64)
        # quarter
        df3['quarter'] = ((df3['month'] - 1) // 3 + 1).astype(np.int64)
        # two months
        df3['2months'] = ((df3['month'] - 1) // 2 + 1).astype(np.int64)
        # Fortnight of Year (1 to 24)
        df3['fortnight_of_year'] = (df3['month'] * 2 - (df3['day'] <= 15).astype(int)).astype(np.int64)
        # Fortnight of Month (0 = first half, 1 = second half)
        df3['fortnight_of_month'] = (df3['day'] > 15).astype(np.int64)

        # week of year
        # "weekofyear" deprecated, in favor of "isocalendar"
        # REF: https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.Series.dt.weekofyear.html
        # REF: https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.Series.dt.isocalendar.html#pandas.Series.dt.isocalendar
        df3['week_of_year'] = df3['date'].dt.isocalendar().week.astype( np.int64 )

    def prepare_assortment_string(self, df3):
        # assortment: a=basic, b=extra, c=extended
        assortment_map = {'a': 'basic', 'b': 'extra', 'c': 'extended'}
        df3['assortment'] = df3['assortment'].map(assortment_map)

    def prepare_state_holiday(self, df3):
        # a=public holiday, b=Easter holiday, c=Christmas, 0=regular working day
        state_holiday_map = {
            'a': 'public_holiday',
            'b': 'easter_holiday',
            'c': 'christmas'
        }
        df3['state_holiday'] = df3['state_holiday'].map(state_holiday_map).fillna('regular_day')

#    def prepare_sales_per_customer(self, df3):
#        df3['sales_per_customer'] = (df3['sales'] / df3['customers']).fillna(0)


    #.......... DATA PREPARATION AUXILIARY METHODS

#    def prepare_response_variable( self, df6 ):
#        df6['sales'] = np.log1p( df6['sales'] )

    def fit_promo_time_week( self, df6 ):
        # apply scaler to 'promo2_time_week' feature
        df6['promo2_time_week'] = self.promo_time_week_scaler.fit_transform( df6[['promo2_time_week']].values )

    def transform_promo_time_week( self, df6 ):
        df6['promo2_time_week'] = self.promo_time_week_scaler.transform( df6[['promo2_time_week']].values )

    def fit_competition_distance( self, df6 ):
        # Numerical variables: Competition Distance & competition_time_month
        # Géron, pg.76: "...before you scale the feature, you should...
        #               "...first transform it to shrink the heavy tail"
        # Apply LOG
        df6['competition_distance'] = np.log1p( df6['competition_distance'] )
        # Then apply MIN-MAX scaling
        df6['competition_distance'] = self.competition_distance_scaler.fit_transform( df6[['competition_distance']].values )

    def transform_competition_distance( self, df6 ):
        # Apply LOG
        df6['competition_distance'] = np.log1p( df6['competition_distance'] )
        # Then apply MIN-MAX scaling
        df6['competition_distance'] = self.competition_distance_scaler.transform( df6[['competition_distance']].values )

    def fit_competition_time_month( self, df6 ):
        # Numerical variables: Competition Distance & competition_time_month
        # Géron, pg.76: "...before you scale the feature, you should...
        #               "...first transform it to shrink the heavy tail"
        # Apply LOG
        df6['competition_time_month'] = np.log1p( df6['competition_time_month'] )
        # Then apply scaler
        df6['competition_time_month'] = self.competition_time_month_scaler.fit_transform( df6[['competition_time_month']].values )

    def transform_competition_time_month( self, df6 ):
        # Apply LOG
        df6['competition_time_month'] = np.log1p( df6['competition_time_month'] )
        # Then apply scaler
        df6['competition_time_month'] = self.competition_time_month_scaler.transform( df6[['competition_time_month']].values )

    def fit_state_holiday( self, df6 ):
        # STATE_HOLIDAY: one_hot_encoder
        ohe1 = self.state_holiday_scaler.fit_transform(df6[['state_holiday']])
        df6.drop(columns=['state_holiday'], inplace=True)
        df6[ohe1.columns] = ohe1
        #df6 = pd.concat([df6, ohe1], axis=1).drop(columns=['state_holiday'])

    def transform_state_holiday( self, df6 ):
        # STATE_HOLIDAY: one_hot_encoder
        ohe1 = self.state_holiday_scaler.transform(df6[['state_holiday']])
        df6.drop(columns=['state_holiday'], inplace=True)
        df6[ohe1.columns] = ohe1
        #df6 = pd.concat([df6, ohe1], axis=1).drop(columns=['state_holiday'])

    def fit_store_type( self, df6 ):
        # Categorical feature: STORE_TYPE - label encoding
        df6['store_type'] = self.store_type_scaler.fit_transform( df6['store_type'] ).astype( np.int64 )

    def transform_store_type( self, df6 ):
        # Categorical feature: STORE_TYPE - label encoding
        df6['store_type'] = self.store_type_scaler.transform( df6['store_type'] ).astype( np.int64 )

    def prepare_assortment_num( self, df6 ):
        # Categorical feature: ASSORTMENT (basic, extra, extended) - Ordinal encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df6['assortment'] = df6['assortment'].map( assortment_dict )

    def fit_year( self, df6 ):
        # year
        df6['year'] = self.year_scaler.fit_transform( df6[['year']].values )

    def transform_year( self, df6 ):
        # year
        df6['year'] = self.year_scaler.transform( df6[['year']].values )

    def prepare_time_features( self, df6 ):
        # month
        df6['month_sin'] = df6['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
        df6['month_cos'] = df6['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )
        # day
        df6['day_sin'] = df6['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/31 ) ) )
        df6['day_cos'] = df6['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/31 ) ) )
        # week_of_year
        df6['week_of_year_sin'] = df6['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/52.5 ) ) )
        df6['week_of_year_cos'] = df6['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/52.5 ) ) )
        # day_of_week
        df6['day_of_week_sin'] = df6['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
        df6['day_of_week_cos'] = df6['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )
        # quarter
        df6['quarter_sin'] = df6['quarter'].apply( lambda x: np.sin( x * ( 2. * np.pi/4 ) ) )
        df6['quarter_cos'] = df6['quarter'].apply( lambda x: np.cos( x * ( 2. * np.pi/4 ) ) )
        # 2-months
        df6['2months_sin'] = df6['2months'].apply( lambda x: np.sin( x * ( 2. * np.pi/6 ) ) )
        df6['2months_cos'] = df6['2months'].apply( lambda x: np.cos( x * ( 2. * np.pi/6 ) ) )
        # fortnight_of_year
        df6['fortnight_of_year_sin'] = df6['fortnight_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/4 ) ) )
        df6['fortnight_of_year_cos'] = df6['fortnight_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/4 ) ) )


class Rossmann( object ):
    def __init__( self ):

        # Conversion object
        self.MyDataConversion = DataConversion()

        # Read scalers/encoders' files
        self.home_path = 'c:/MeusEstudos/CURSOS TI/Em 2023 - ComunidadeDS/Projetos do Aluno/PA.04 Rossmann/webapp/'
        self.competition_distance_scaler   = pickle.load( open( self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb' ) )
        self.competition_time_month_scaler = pickle.load( open( self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb' ) )
        self.promo_time_week_scaler        = pickle.load( open( self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb' ) )
        self.state_holiday_scaler          = pickle.load( open( self.home_path + 'parameter/state_holiday_scaler.pkl', 'rb' ) )
        self.store_type_scaler             = pickle.load( open( self.home_path + 'parameter/store_type_scaler.pkl', 'rb' ) )
        self.year_scaler                   = pickle.load( open( self.home_path + 'parameter/year_scaler.pkl', 'rb' ) )

        # Set scalers/encoders to DataConversion object (to be used in TRANSFORM)
        self.MyDataConversion.set_scalers({
            'promo_time_week':        self.promo_time_week_scaler,
            'competition_distance':   self.competition_distance_scaler,
            'competition_time_month': self.competition_time_month_scaler,
            'store_type':    self.store_type_scaler,
            'state_holiday': self.state_holiday_scaler,
            'year':          self.year_scaler
        })

    #........... Class Methods
    def data_cleaning(self, df1):
        # Copied from section 1, items 1.1, 1.3 and 1.4

        # 1.1. RENAME COLUMNS
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                    'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval' ]
        snakecase = lambda x: inflection.underscore( x )
        cols_new = list( map( snakecase, cols_old ) )
        # rename
        df1.columns = cols_new

        # 1.3. Data Types
        df1['date'] = pd.to_datetime( df1['date'] )

        # 1.3. FILLOUT NA

        # competition_distance
        df1['competition_distance'] = df1['competition_distance'].fillna(200000.0)
        # competition_open_since_month
        df1['competition_open_since_month'] = df1['competition_open_since_month'].fillna(df1['date'].dt.month)
        # competition_open_since_year
        df1['competition_open_since_year'] = df1['competition_open_since_year'].fillna(df1['date'].dt.year)
        # promo2_since_week
        df1['promo2_since_week'] = df1['promo2_since_week'].fillna(df1['date'].dt.isocalendar().week)
        # promo2_since_year
        df1['promo2_since_year'] = df1['promo2_since_year'].fillna(df1['date'].dt.year)
        # promo_interval
        df1['promo_interval'].fillna( 0, inplace=True )

        # 1.4. CHANGE TYPES
        # From FLOAT to INT
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( np.int64 )
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype( np.int64 )
        df1['promo2_since_week']          = df1['promo2_since_week'].astype( np.int64 )
        df1['promo2_since_year']         = df1['promo2_since_year'].astype( np.int64 )

        return df1

    def feature_engineering(self, df3):

        # Copied from section 3, item 3.2
        self.MyDataConversion.feature_engineering(df3)

        # 4.1. ROWS FILTERING
        df3 = df3[(df3['open'] != 0)]
#        df3 = df3[(df3['open'] != 0) & (df3['sales'] > 0)]

        # 6.1. COLUMNS FILTERING
        cols_drop = [ 'open', 'promo_interval', 
                    #  'customers', 'sales_per_customer', 
                      'promo2_since', 'competition_since' ]
        df3 = df3.drop( cols_drop, axis=1 )

        return df3

    def data_preparation(self, df5):

        self.MyDataConversion.data_preparation_transform( df5 )

        # 7.3. Features from Boruta
        cols_selected = [
            'store', 
            'day_of_week', 
            'promo', 
            'store_type', 
            'assortment',
            'competition_distance', 
            'competition_open_since_month', 
            'competition_open_since_year',
            'promo2', 
            'promo2_since_week', 
            'promo2_since_year', 
            'day', 
            'week_of_year',
            'promo2_time_week', 
            'competition_time_month', 
            'day_sin', 
            'day_cos', 
            'day_of_week_sin', 
            'day_of_week_cos'
        ]
        return df5[ cols_selected ]

    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict( test_data )

        # join pred into the original data
        original_data['prediction'] = np.expm1( pred )

        return original_data.to_json( orient='records', date_format='iso' )
