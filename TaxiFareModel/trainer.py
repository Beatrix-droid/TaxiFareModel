#imports from custom modules
from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
from data import get_data, clean_data
from encoders import DistanceTransformer, TimeFeaturesEncoder
from utils import compute_rmse


#sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Trainer():
    def __init__(self, X, y):

        """
            X: pandas DataFrame
            y: pandas Series

        """
        #fetching data
        df = get_data(nrows=10_000)

        #cleaning it
        clean_data(df, test=False)

        #preparing the features and the target
        self.pipeline = None
        self.y = df.pop("fare_amount")
        self.X = df

        #holdout:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)



    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return pipe

    def run(self):

        # train the pipelined model
        self.set_pipeline.fit(self.X_train, self.y_train)
        return self.set_pipeline


    def evaluate (self, X_test, y_test):
        '''returns the value of the RMSE'''

        self.y_pred = self.set_pipeline.predict(self.X_test)

        rmse = compute_rmse(self.y_pred, self.y_test)
        print(rmse)
        return rmse

        """evaluates the pipeline on df_test and return the RMSE"""



if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
