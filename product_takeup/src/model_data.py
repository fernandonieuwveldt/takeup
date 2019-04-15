import feather
from sklearn.model_selection import train_test_split
from os import path


class ProductTakeupData(object):
    """
    Simple data object to store all training data information

    Args:
        file_name: feather file name containing data

    Attributes:
        data: feather dataframe containing all training data
        features: features of the training data
        target: Target variable, client that took up product
        X_train: the data on which model will be trained on
        X_test: the data on which the model will be tested on
        y_train: target variable for training
        y_test: target variable for testing

    """
    def __init__(self, file_name=''):
        self.file_name = file_name
        self.data = feather.read_dataframe(self.file_name)
        self.features = self.data.drop('SS', axis=1)
        self.target = self.data['SS'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.target,
                                                                                stratify=self.target, random_state=42)


if __name__ == '__main__':
    data_folder = path.join(path.dirname(path.realpath(__file__)), '../data/')
    data = ProductTakeupData(file_name='%s/train_data' % data_folder)
