from xgboost import DMatrix
import pickle
import matplotlib.pyplot as plt


def train_valid_test(path, dmat=True):
    train, valid, test, task = pickle.load(open(path, 'rb'))
    if dmat:
        train, valid, test = DMatrix(*train), DMatrix(*valid), DMatrix(*test)
    return train, valid, test, task


def train_test(path='traintest', dmat=True):
    # after pickle.dump(((new_train, train_target), (eval_matrix, eval_target)), open('traintest', 'wb'))
    train, test = pickle.load(open(path, 'rb'))
    if dmat:
        train, test = DMatrix(*train), DMatrix(*test)
    return train, test


def plot_comparison(callbacks):
    plt.figure(figsize=(16,9))
    try:
        for callback in callbacks:
            callback.log.plot(ax=plt.gca())
    except TypeError:
        callbacks.log.plot(ax=plt.gca())
    plt.legend()
    plt.show()


from collections import OrderedDict, Callable


class OrderedDefaultDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))