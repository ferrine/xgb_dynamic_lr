from xgboost import DMatrix
import pickle


def train_test():
    # after pickle.dump(((new_train, train_target), (eval_matrix, eval_target)), open('traintest', 'wb'))
    train, test = pickle.load(open('traintest', 'rb'))
    train, test = DMatrix(*train), DMatrix(*test)
    return train, test