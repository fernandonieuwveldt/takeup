import learn
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import model_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.externals import joblib
from os import path


def check_fitted(clf):
    """
    Checks if a sklearn model was fitted

    :param clf: classifier
    :return: True if model was fitted, False if not
    """
    return hasattr(clf, "classes_")


def predict(testdata, trained_model):
    """
    Predicts unseen data using a trained model

    :param testdata: Data that the model will be tested on
    :param trained_model: Model that was already trained
    :return: tuple containing class probabilities and classes
    """
    if check_fitted(trained_model):
        score = trained_model.predict_proba(testdata)[:, 1]
        predictions = trained_model.predict(testdata)
        return score, predictions
    else:
        print 'model should be fitted first'


def model_check(scores, y_true):
    """
    Checks how well our model works on test set using a ROC curve

    :param scores: class probabilities from model
    :param y_true: true labels
    :return: ROC curve plot with AUC value
    """
    auc = round(roc_auc_score(y_true, scores), 3)
    fpr, tpr, threshold = roc_curve(y_true, scores)
    plt.plot(fpr, tpr, label="AUC on Test set =" + str(auc))
    plt.plot([[0, 0], [1, 1]])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.grid()
    plt.show()


def save_model(clf, name='model'):
    """
    Save trained model for offline use

    :param clf: trained classifier
    :param name: file name it will be saved to
    :return: saved pickle file
    """
    model_folder = path.join(path.dirname(path.realpath(__file__)), '../saved_models/')
    pickle_file = '%s.joblib' % name
    joblib.dump(clf, '%s/%s' % (model_folder, pickle_file), compress=9)


def load_model(model_name=''):
    """
    Loads saved pickle file of trained model
    :param model_name: pickle file to be loaded
    :return: a loaded model
    """
    model_folder = path.join(path.dirname(path.realpath(__file__)), '../saved_models/')
    return joblib.load('%s/%s' % (model_folder, model_name))


def apply_saved_models(data_to_predict=None):
    import sys
    sys.path.append(path.dirname(path.realpath(__file__)))
    logistic_regression = load_model('LogisticRegression.joblib')
    random_forest = load_model('RandomForestClassifier.joblib')
    scores_lr, _ = predict(data_to_predict, logistic_regression)
    scores_rf, _ = predict(data_to_predict, random_forest)
    return {'LogisticRegression_score': scores_lr, 'RandomForestClassifier_score': scores_rf}


if __name__ == '__main__':
    data = model_data.ProductTakeupData(file_name='../data/train_data')
    # data = feather.read_dataframe('../data/train_data')
    # X = data.drop('SS', axis=1)
    # y = data['SS'].values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    # trained_model = learn.model_fit(X_train, y_train)
    # scores, classes = predict(X_test, trained_model)
    # loaded_model = load_model('logistic_regression.pkl')
    # scores, classes = predict(X_test, loaded_model)
    clf = LogisticRegression()
    # grid = {'classifier__penalty': ['l1', 'l2'], 'classifier__C': 1/np.linspace(0.001, 1, 10)}
    # clf = LogisticRegression()
    # bc = BaggingClassifier(base_estimator=clf, n_estimators=300, n_jobs=-1)
    # rf_grid = {'classifier__n_estimators': [150], 'classifier__max_features': [range(2, 12, 2)]}
    # rf = RandomForestClassifier(n_estimators=150)
    trained_model = learn.model_fit(data.X_train, data.y_train, classifier=clf)
    scores, classes = predict(data.X_test, trained_model)
    save_model(trained_model, 'LogisticRegression')
    model_check(scores, data.y_test)
    print apply_saved_models(data_to_predict=data.X_test.head(1))
