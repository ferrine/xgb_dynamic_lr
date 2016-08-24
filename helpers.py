from xgboost import DMatrix
import pickle


def train_test(dmat=True):
    # after pickle.dump(((new_train, train_target), (eval_matrix, eval_target)), open('traintest', 'wb'))
    train, test = pickle.load(open('traintest', 'rb'))
    if dmat:
        train, test = DMatrix(*train), DMatrix(*test)
    return train, test


def traincb(callbacks_to_test, params, train, test):
    import xgboost as xgb
    gbms = []
    import sys
    # support class to redirect stderr
    class flushfile(object):
        def __init__(self, f):
            self.f = f

        def __getattr__(self, name):
            return object.__getattribute__(self.f, name)

        def write(self, x):
            self.f.write(x)
            self.f.flush()

        def flush(self):
            self.f.flush()

    oldstdout = sys.stdout
    for i, cbs in enumerate(callbacks_to_test):
        sys.stdout = open("xgb_" + str(i) + "_log.txt", "w")
        sys.stdout = flushfile(sys.stdout)
        gbms.append(
            xgb.train(dtrain=train, callbacks=cbs, params=params, num_boost_round=1000, early_stopping_rounds=15,
                      verbose_eval=True, evals=[(test, "val_0")]))

    sys.stdout = oldstdout
    return gbms


def plot_comparison(callbacks_to_test):
    import matplotlib.pyplot as plt

    def parse_xgb_log(fname):
        iterations = []
        accuracy = []
        for line in open(fname):
            if line.startswith("["):
                iterations.append(int(line.split("[")[1].split("]")[0]))
                accuracy.append(float(line.split(":")[1].strip()))
        return iterations, accuracy

    for i, cb in enumerate(callbacks_to_test):
        if len(cb) < 3:
            cb_name = "default"
        else:
            cb_name = str(cb[0]).split()[1].split(".")[0]
        iterations, accuracy = parse_xgb_log("xgb_" + str(i) + "_log.txt")
        plt.plot(accuracy, label=cb_name)

    plt.legend()
    plt.show()