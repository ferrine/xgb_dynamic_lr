from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from functools import wraps
import collections
import pickle
import warnings
import os
import re
import six

from xgboost.core import EarlyStopException
from objectives import dlinear
import numpy as np
import pandas as pd
from helpers import OrderedDefaultDict

VAL = re.compile(r'(?P<name>[^:]+):(?P<val>\d+(?:\.\d+)?)')


class BaseLRMeta(ABCMeta):
    """"метакласс нужен для сокращения кода"""
    def __call__(cls, *args, **kwargs):
        id_ = next(cls._counter)
        base_dir = 'results'
        if not os.path.exists(os.path.join(base_dir, cls.__name__)):
            os.mkdir(os.path.join(base_dir, cls.__name__))

        instance = super(BaseLRMeta, cls).__call__(*args, **kwargs)
        instance.tag = os.path.join(base_dir, cls.__name__, 'trace_%d' % id_)
        instance.configdir = os.path.join(base_dir, cls.__name__, 'config_%d' % id_)

        evals = kwargs.pop('evals', [])
        names = map(lambda t: t[1], evals)
        js = {'args': args, 'kwargs': kwargs}

        with open(instance.tag, 'w') as f:
            f.write('\t'.join(names))
            f.write('\n')

        try:
            with open(instance.configdir, 'wb') as f:
                pickle.dump(js, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        except pickle.PicklingError:
            warnings.warn('cannot pickle %s' % instance.configdir)
        return instance

    def __new__(mcs, name, bases, dic):
        # вот тут мы просто собираем класс, это то,
        # что пришло после объявления директивы class
        cls = super(BaseLRMeta, mcs).__new__(mcs, name, bases, dic)
        # это то, что пришло после ее исполнения, после class MyClass: ...
        # напишем простенькие врапперы для сразу всех классов наследников

        def integers(start=0):
            i = start
            while True:
                yield i
                i += 1

        cls._counter = integers()

        def wrap_init(__init__):
            @wraps(__init__)
            def wrapped(self, *args, evals=(), feval=None, **kwargs):
                __init__(self, *args, **kwargs)
                # чтобы тыщу раз не вызывать аля super(MyRate, self).__init__()
                # или эту строчку
                self._name = name
                self._inited = False
                self.feval = feval
                self.evals = evals
                self.trace = OrderedDefaultDict(list)
            return wrapped

        def wrap_call(__call__):
            cast = lambda t: (t[0].strip(), float(t[1]))
            @wraps(__call__)
            def wrapped(self, env):
                # чтобы тыщу раз не прописывать это условие
                # ну и продолжаем полет
                msg = str(env.model.eval_set(self.evals, env.iteration, self.feval))
                res = collections.OrderedDict(map(cast, VAL.findall(str(msg))))
                for key, val in res.items():
                    self.trace[key].append(val)
                if res:
                    with open(self.tag, 'a') as f:
                        f.write('\t'.join(map(str, res.values())))
                        f.write('\n')
                if not self._inited:
                    self._init(env)
                    self._inited = True
                __call__(self, env)
            return wrapped

        # враппим
        cls.__init__ = wrap_init(cls.__init__)
        cls.__call__ = wrap_call(cls.__call__)
        return cls


class BaseLR(six.with_metaclass(BaseLRMeta)):
    def _init(self, env):
        """If it is first call this method will be called
        override it for your needs, by default does nothing
        """

    @abstractmethod
    def __call__(self, env):
        """Write callback logic here, be sure that init is already called
        """

    @property
    def log(self):
        path = self.tag
        df = pd.read_csv(path, sep='\t', prefix=self.tag.replace('/', ':'))
        f = lambda x: '{}@{}'.format(
                self.tag, x,
        )
        df.rename(columns=f, inplace=True)
        return df

    @property
    def config(self):
        return pickle.load(open(self.configdir, 'rb'))

    def __hash__(self):
        return hash(self.tag)


class DoNothing(BaseLR):
    def __call__(self, env):
        "bugaga"


class DynamicLR(BaseLR):
    def __init__(self, start_lr, min_lr, decrease_function, rounds_function):
        self.cur_lr = start_lr
        self.best_iteration = 0
        self.best_score = float('inf')
        self.failed_iter = 0
        self.decrease_count = 0
        self.min_lr = min_lr
        self.decrease_function = decrease_function
        self.rounds_function = rounds_function
        
    def _init(self, env):
        bst = env.model
        if len(self.trace.values()) == 0:
            raise ValueError('For LR-based early stopping you need at least one set in evals.')
        if bst is not None:
            if bst.attr('best_score') is not None:
                self.best_score = float(bst.attr('best_score'))
                self.best_iteration = int(bst.attr('best_iteration'))
                self.best_msg = bst.attr('best_msg')
            else:
                bst.set_attr(best_iteration=str(self.best_iteration))
                bst.set_attr(best_score=str(self.best_score))
        else:
            assert env.cvfolds is not None
        bst.set_param("learning_rate", self.start_lr)
    
    def __call__(self, env):
        score = list(self.trace.values())[0][-1]
        best_score = self.best_score
        best_iteration = self.best_iteration

        if score < best_score:
            self.best_score = score
            self.best_iteration = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(best_score=str(self.best_score),
                                   best_iteration=str(self.best_iteration))

        elif env.iteration - best_iteration >= self.rounds_function(self.decrease_count):
            if self.failed_iter < self.rounds_function(self.decrease_count):
                self.failed_iter += 1
            else:
                env.model.set_param("learning_rate", self.decrease_function(self.cur_lr))
                self.cur_lr = self.decrease_function(self.cur_lr)
                print("lowered lr", self.cur_lr)
                if self.cur_lr < self.min_lr:
                    print("Reached minimal LR, stopped. LR:", self.cur_lr, "Score:", self.best_score)
                    raise EarlyStopException(best_iteration)

                self.failed_iter = 0
                self.decrease_count += 1
    

class BoldDriver(BaseLR):
    def __init__(self, start_lr, min_lr, boldness, timidness, relax, relax_k):
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.boldness = boldness
        self.timidness = timidness
        self.relax = relax
        self.relax_k = relax_k
    
    def _init(self, env):
        bst = env.model

        if len(self.trace.values()) == 0:
            raise ValueError('For LR-based early stopping you need at least one set in evals.')

        self.cur_lr = self.start_lr
        self.best_iteration = 0
        self.prev_score = float('inf')
        self.relaxation_rounds = 0
        bst.set_param("learning_rate", self.start_lr)
    
    def __call__(self, env):
        score = list(self.trace.values())[0][-1]
        prev_score = self.prev_score
        best_iteration = self.best_iteration

        if score < prev_score:
            self.prev_score = score
            self.best_iteration = env.iteration
            self.cur_lr *= self.boldness
            env.model.set_param("learning_rate", self.cur_lr)
        else:

            self.prev_score = score
            if self.relaxation_rounds < 0:
                print("Reduced LR from", self.cur_lr, "to", self.cur_lr * self.timidness)
                self.relaxation_rounds = self.relax * self.relax_k
                self.relax = self.relaxation_rounds
                self.cur_lr *= self.timidness

            else:
                self.relaxation_rounds -= 1
            env.model.set_param("learning_rate", self.cur_lr)
            if self.cur_lr < self.min_lr:
                print("Reached minimal LR, stopped. LR:", self.cur_lr, "Last Score:", self.prev_score)
                raise EarlyStopException(best_iteration)


class McClain(BaseLR):
    def __init__(self, start_lr, target_lr):
        self.start_lr = start_lr
        self.target_lr = target_lr
    
    def _init(self, env):
        bst = env.model

        if len(env.evaluation_result_list) == 0:
            raise ValueError('For LR-based early stopping you need at least one set in evals.')

        self.cur_lr = self.start_lr
        self.prev_score = float('inf')
        bst.set_param("learning_rate", self.start_lr)

    def __call__(self, env):
        prev_lr = self.cur_lr
        lr = prev_lr / (prev_lr + 1 - self.target_lr)
        self.cur_lr = lr
        env.model.set_param("learning_rate", lr)


class Stc(BaseLR):
    def __init__(self, start_lr, T):
        self.start_lr = start_lr
        self.T = T

    def _init(self, env):
        bst = env.model
        self.start_lr = self.start_lr
        self.best_iteration = 0
        bst.set_param("learning_rate", self.start_lr)

    def __call__(self, env):
        lr = self.start_lr / (1 + env.iteration / self.T)
        env.model.set_param("learning_rate", lr)


class GradBased(BaseLR):
    def __init__(self, trainset, grads=dlinear, howto=np.mean):
        self.trainset = trainset
        self.true = trainset.get_label()
        self.grads = grads
        self.howto = howto  # the way to calculate gradient for function, returnes scalar

    def grad(self, env):
        # get gradient wrt function
        bst = env.model
        pred = bst.predict(self.trainset)
        grads = self.grads(pred, self.true)
        grad = float(self.howto(grads))
        return grad


class TrackGradMean(GradBased):
    def __init__(self, trainset, grads=dlinear, howto=np.mean, e=1e-8, b1=0.9, b2=0.999, a=1e-3):
        super(TrackGradMean, self).__init__(trainset, grads, howto)
        self.rmean = 0
        self.rmean2 = 0
        self.e = e
        self.b1 = b1
        self.b2 = b2
        self.b1t=b1
        self.b2t=b2
        self.a = a

    def __call__(self, env):
        grad = self.grad(env)
        grad2 = grad**2
        self.rmean = self.b1 * self.rmean + (1 - self.b1) * grad
        self.rmean2 = self.b2 * self.rmean2 + (1 - self.b2) * grad2

        lr = self.a*(np.abs(self.rmean)/(1-self.b1t)) / (np.sqrt(self.rmean2/(1-self.b2t)) + self.e)
        print(lr)
        self.b1t *= self.b1
        self.b2t *= self.b2
        env.model.set_param("learning_rate", lr)


class PredictLoss(BaseLR):
    def __init__(self, hist=30, posmax=15, lr=0.2):
        from sklearn.linear_model.base import LinearRegression
        from collections import deque
        self.hist = hist
        self.track = deque(maxlen=self.hist)
        self.regr = LinearRegression()
        self.poscases = 0
        self.posmax = posmax
        self.lr = lr

    def __call__(self, env):
        if len(self.track) > 5:
            y = np.array(self.track)
            x = np.array(range(len(y.shape))).reshape(-1, 1)
            self.regr.fit(x, y)
            coef_ = self.regr.coef_[0]
            preds = self.regr.predict(x)
            fst = preds[0]
            lst = preds[-1]
            e = np.sqrt(((y - preds)**2).mean())
            if coef_ > 0:
                self.poscases += 1
                if self.poscases >= self.posmax:
                    raise EarlyStopException
            else:
                self.poscases -= 1
                if self.poscases < 0:
                    self.poscases = 0
            diff = np.abs(fst - lst)
            coef = np.clip(diff/e, 1e-6, 1)
            lr = self.lr*coef
            env.model.set_param("learning_rate", lr)

