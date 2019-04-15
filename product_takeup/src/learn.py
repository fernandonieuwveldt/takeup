from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid


class DropNaTransformer(BaseEstimator, TransformerMixin):
    """
    Drops observation(row) if there are too many missing values
    """
    def __init__(self, nr_nans=0):
        self.nr_nans = nr_nans

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.nr_nans > 0:
            idx_nans = X.isnull().sum(axis=1) >= self.nr_nans
            return X[~idx_nans]
        else:
            return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that select columns based on type or from list of names
    """
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(self.features, str):
            return X.select_dtypes(include=self.features)
        elif isinstance(self.features, list):
            return X[self.features]
        else:
            return X


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # this is where one can do feature engineering
        return X


def pipeline(classifier, param_grid={}):
    """
    :param classifier: sklearn classifier object
    :param param_grid: parameter to search over in GridSearch
    :return: a sklearn GridSearch object
    """
    # drop_na = Pipeline(steps=[('drop_nans', DropNaTransformer(nr_nans=16))])
    # drop_na = DropNaTransformer(nr_nans=16)
    categorical_features = ['LoanClient', 'InactiveLoanClient', 'DormantLoanClient', 'LoanEver']
    categorical_pipeline = Pipeline(steps=[('categorical_selector', FeatureSelector(categorical_features)),
                                           ('categorical_transformer', CategoricalTransformer())])

    numerical_features = ['Inflows_Total', 'Outflows_Total_L3M', 'Inflows_Above_1k',
                          'Max_Dep_Bal_L3M', 'Other_Perc_L3M', 'Outflow_Max_L3M',
                          'Inflows_Max_AVG_Day', 'Ave_Days_Above_100_L3M', 'CW_Perc_L3M',
                          'DO_Perc_L3M', 'DODispute_L3M', 'Val_POS_L3M', 'Val_DO_L3M',
                          'Avg_Dep_Bal_L3M', 'CSWEEP_P90_L3M', 'CW_Util_L3M']

    numerical_pipeline = Pipeline(steps=[('numerical_selector', FeatureSelector(numerical_features)),
                                         ('numerical_transformer', NumericalTransformer()),
                                         ('imputer', SimpleImputer(strategy='median')),
                                         ('scaler', StandardScaler())])
    # combine transform pipelines
    transformer_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
                                                          ('numerical_pipeline', numerical_pipeline)])
    # combine transformer and estimator to create full pipeline
    estimator = Pipeline(steps=[('transformer_pipeline', transformer_pipeline),
                                ('classifier', classifier)])
    n_combo = len(ParameterGrid(param_grid))
    if not param_grid:
        return estimator
    else:
        if n_combo > 10:
            print 'using RandomizedSearch'
            return RandomizedSearchCV(estimator, param_distributions=param_grid, cv=5, n_jobs=-1)
        else:
            # not enough combinations for random search, use grid search instead
            print 'using GridSearchCV'
            return GridSearchCV(estimator, param_grid=param_grid, cv=5, n_jobs=-1)


def model_fit(X, y, classifier=LogisticRegression(), param_grid={}):
    """
    :param X: input data to fit model on
    :param y: training labels
    :param classifier: classifier to be fit model, LogisticRegression by default
    :param param_grid: parameter grid if a grid search needs to be applied, empty for now grid search
    :return: model trained with the pipeline
    """
    full_pipeline = pipeline(classifier, param_grid)
    return full_pipeline.fit(X, y)


if __name__ == '__main__':
    import model_data
    # data = feather.read_dataframe('../data/train_data')
    # print 'number of rows before dropping nans: ', data.shape
    # drop_na = DropNaTransformer(nr_nans=16)
    # drop_na.fit(data)
    # data = drop_na.transform(data)
    # print 'number of rows after dropping nans: ', data.info()
    # X = data.drop('SS', axis=1)
    # y = data['SS'].values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    data = model_data.ProductTakeupData(file_name='../data/train_data')
    trained_model = model_fit(data.X_train, data.y_train)
    ypreds = trained_model.predict_proba(data.X_test)[:, 1]
    print roc_auc_score(data.y_test, ypreds)

