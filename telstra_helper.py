# ----------------------------------------------------------
# Helper Module for 'Telstra Data Disruptions'
#
# Author: Chieko N.
# -----------------------------------------------------------
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import confusion_matrix

def check_bad_col(df, feature):
    """ Check if the DataFrame contains invalid data format. """

    bad_rows_id = df.loc[df["id"].str.isnumeric() == False]
    bad_rows_type = df.loc[df[feature].str.contains(feature, regex=False) == False]

    # return pandas.Series of innapropriate rows
    return pd.concat([bad_rows_id, bad_rows_type]).drop_duplicates()

def read_tables():
    """ Read csv files and return tables in a dictionary."""

    tables = {}

    ### Main datafile
    print("----- train.csv -----")
    data_main = pd.read_csv("data/train.csv")
    display(data_main.head(n=5))
    display(data_main.dtypes)

    ### Log feature
    print("\n----- log_feature.csv -----")
    log_feature = pd.read_csv("data/log_feature.csv")
    display(log_feature.head(n=5))
    display(log_feature.dtypes)

    ### Severity type
    print("\n----- severity_type.csv -----")
    severity_type = pd.read_csv("data/severity_type.csv")
    display(severity_type.head(n=5))
    display(severity_type.dtypes)

    ### Event type
    print("\n----- event_type.csv -----")
    event_type_raw = pd.read_csv("data/event_type.csv", header=None,
                                names=["id", "event_type", "event_flag"])
    # Delete header
    event_type = event_type_raw.drop(labels=0, axis=0)
    # display(event_type.dtypes)

    # Detect badly formatted rows
    bad_rows_et = check_bad_col(event_type, "event_type").index.values
    print("\nevent_type.csv :Number of wrong rows = {}, positions = {}\n".format(len(bad_rows_et), bad_rows_et))
    # Delete bad rows
    event_type = event_type.drop(bad_rows_et, axis=0)
    # Cast id into numeric
    event_type["id"] = pd.to_numeric(event_type["id"], errors='coerce')
    display(event_type.head(n=5))
    display(event_type.dtypes)


    ### Resource type
    print("\n----- resource_type.csv -----")
    resource_type_raw = pd.read_csv("data/resource_type.csv", header=None,
                                    names=["id", "resource_type", "resource_flag"])
    # Delete header
    resource_type = resource_type_raw.drop(labels=0, axis=0)
    # display(resource_type.dtypes)

    # Detect badly formatted rows
    bad_rows_rt = check_bad_col(resource_type, "resource_type").index.values
    print("\nresource_type.csv: Number of wrong rows = {}, positions = {}\n".format(len(bad_rows_rt), bad_rows_rt))
    # Delete bad rows
    resource_type = resource_type.drop(bad_rows_rt, axis=0)
    # Cast id into numeric
    resource_type["id"] = pd.to_numeric(resource_type["id"], errors='coerce')
    display(resource_type.head(n=5))
    display(resource_type.dtypes)

    tables['train.csv'] = data_main
    tables['log_feature.csv'] = log_feature
    tables['severity_type.csv'] = severity_type
    tables['event_type.csv'] = event_type
    tables['resource_type.csv'] = resource_type

    return tables


def getint(s):
    """ Get feature No. (int) from string.
        for example) 'resource_type 123' ---> int(123)
    """
    return int(s.split()[-1])


def getint_list(l):
    """ Get feature No.s (int) from strings. Input is a list of 'feature No'."""
    return [getint(s) for s in l]


def logloss_confmat(clf, X_train, y_train, X_test, y_test, savefile=None):
    """ Print multi-class log loss and confusion matrix.
        Return: result(dict) = {'log_loss': [train, test], # log loss value
                                'conf_mat': [train, test]  # confusion matrix
                                }
    """

    result = {}
    # predict training data
    pred_train = clf.predict(X_train)
    predproba_train = clf.predict_proba(X_train)
    # predict testing data
    pred_test = clf.predict(X_test)
    predproba_test = clf.predict_proba(X_test)

    result['log_loss'] = [log_loss(y_train, predproba_train), log_loss(y_test, predproba_test)]
    result['conf_mat'] = [confusion_matrix(y_train, pred_train), confusion_matrix(y_test, pred_test)]

    print("\n***** training *****")
    print(" log loss for train = {}".format(result['log_loss'][0]))
    print(" --- confusion matrix ---")
    print(result['conf_mat'][0])
    print("\n***** testing *****")
    print(" log loss for test = {}".format(result['log_loss'][1]))
    print(" --- confusion matrix ---")
    print(result['conf_mat'][1])

    # Show heatmap for testing data with normalization
    cnfmat_test = result['conf_mat'][1]
    classes = ['fault_severity {}'.format(i) for i in range(len(cnfmat_test))]
    cnfmat_test_norm = cnfmat_test.astype('float') / cnfmat_test.sum(axis=1)[:, np.newaxis]
    df_cnfmat_test = pd.DataFrame(cnfmat_test_norm, index=classes, columns=classes)
    fr_heatmap = sns.heatmap(df_cnfmat_test, annot=True, vmin=0, vmax=1, cmap="Blues")
    plt.title("Confusion Matrix for testing data with normalization")

    if savefile != None:
        # Save image into file
        plt.savefig("figures/confmat_{}.png".format(savefile), bbox_inches='tight')

    plt.show()

    return result

def logloss_confmat_dnn(model, X_train_array, y_train_array, \
                        X_test_array, y_test_array, savefile=None):
    """ Print multi-class log loss and confusion matrix for the test.
        DNN version of logloss_confmat.
        Return: result(dict) = {'log_loss': <log loss value>
                                'conf_mat': <confusion matrix>
                                }
    """

    result = {}
    # Predict training and testing data
    # Output of predict() is arrays of probabilities
    predproba_train = model.predict(X_train_array)
    predproba = model.predict(X_test_array)
    # Get a predicted class from probabilities
    pred_train = np.argmax(predproba_train, axis=1)
    pred = np.argmax(predproba, axis=1)
    # Make an array of label
    y_train = np.where(y_train_array == 1)[1]
    y_test = np.where(y_test_array == 1)[1]

    result['log_loss'] = [log_loss(y_train_array, predproba_train), log_loss(y_test_array, predproba)]
    result['conf_mat'] = [confusion_matrix(y_train, pred_train), confusion_matrix(y_test, pred)]

    #score = model.evaluate(X_test_array, y_test_array, verbose=0)

    print("\n***** training *****")
    #print(" Evaluation {} = {}".format(model.metrics_names, score))
    print(" log loss for train = {}".format(result['log_loss'][0]))
    print(" --- confusion matrix ---")
    print(result['conf_mat'][0])
    print("\n***** testing *****")
    #print(" Evaluation {} = {}".format(model.metrics_names, score))
    print(" log loss for test = {}".format(result['log_loss'][1]))
    print(" --- confusion matrix ---")
    print(result['conf_mat'][1])

    # Show heatmap for testing data with normalization
    cnfmat_test = result['conf_mat'][1]
    classes = ['fault_severity {}'.format(i) for i in range(len(cnfmat_test))]
    cnfmat_test_norm = cnfmat_test.astype('float') / cnfmat_test.sum(axis=1)[:, np.newaxis]
    df_cnfmat_test = pd.DataFrame(cnfmat_test_norm, index=classes, columns=classes)
    fr_heatmap = sns.heatmap(df_cnfmat_test, annot=True, vmin=0, vmax=1, cmap="Blues")
    plt.title("Confusion Matrix for testing data with normalization")

    if savefile != None:
        # Save image into file
        plt.savefig("figures/confmat_{}.png".format(savefile), bbox_inches='tight')

    plt.show()

    return result

def print_fi(fi, columns, top=20):
    """
    Print Feature Importance and its proportion in decsending order.
    top: Top N features are listed (default:N=20)
    """

    all_fi = np.sum(fi)
    fi_sorted = np.sort(fi, axis=None)[::-1]
    fi_sorted_index = np.argsort(fi, axis=None)[::-1]

    print("FEATURE: FI VALUE, PERCENT (CUMULATIVE VALUE [PERCENT]) ")
    print("--------------------------------------------------")

    fi_cum = 0
    for i, val in enumerate(fi_sorted):
        fi_cum += val
        print("({}) {}: {}, {:2.2%} ({} [{:2.2%}])".format(i+1, columns[fi_sorted_index[i]], val, \
                                                    val/all_fi, fi_cum, fi_cum/all_fi))
        if i == (top - 1):
            break
