from __future__ import unicode_literals
from xgboost.core import EarlyStopException
from abc import ABCMeta, abstractmethod
from functools import wraps
from objectives import dlinear
import numpy as np
import re
import collections
VAL = re.compile(r'(?P<name>[^:]+):(?P<val>\d+(?:\.\d+)?)')
import pickle
import os
import six

class BaseLRMeta(ABCMeta):
    """"метакласс нужен для сокращения кода"""
    def __call__(cls, *args, **kwargs):
        id_ = cls._counter
        cls._counter += 1
        if not os.path.exists(cls.__name__):
            os.mkdir(cls.__name__)
        instance = super(BaseLRMeta, cls).__call__(*args, **kwargs)
        instance.tag = os.path.join(cls.__name__, 'trace_%d' % id_)
        evals = kwargs.pop('evals', [])
        names = map(lambda t: t[1], evals)
        js = {'args': args, 'kwargs': kwargs}
        configdir = os.path.join(cls.__name__, 'config_%d' % id_)
        if not os.path.exists(cls.__name__):
            os.mkdir(cls.__name__)
        with open(instance.tag, 'w') as f:
            f.write('\t'.join(names))
            f.write('\n')
        with open(configdir, 'wb') as f:
            pickle.dump(js, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        return instance

    def __new__(mcs, name, bases, dic):
        # вот тут мы просто собираем класс, это то,
        # что пришло после объявления директивы class
        cls = super(BaseLRMeta, mcs).__new__(mcs, name, bases, dic)
        # это то, что пришло после ее исполнения, после class MyClass: ...
        # напишем простенькие врапперы для сразу всех классов наследников
        cls._counter = 0

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
                self.trace = collections.defaultdict(list)
            return wrapped

        def wrap_call(__call__):
            @wraps(__call__)
            def wrapped(self, env):
                # чтобы тыщу раз не прописывать это условие
                if not self._inited:
                    self._init(env)
                    self._inited = True
                # ну и продолжаем полет
                cast = lambda t: (t[0].strip(), float(t[1]))
                msg = str(env.model.eval_set(self.evals, env.iteration, self.feval))
                res = collections.OrderedDict(map(cast, VAL.findall(str(msg))))
                for key, val in res.items():
                    self.trace[key].append(val)
                if res:
                    with open(self.tag, 'a') as f:
                        f.write('\t'.join(map(str, res.values())))
                        f.write('\n')
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


def dynamic_lr(start_lr, min_lr, decrease_function, rounds_function):
    state = {}

    def init(env):
        bst = env.model

        if len(env.evaluation_result_list) == 0:
            raise ValueError('For LR-based early stopping you need at least one set in evals.')

        state["cur_lr"] = start_lr
        state['best_iteration'] = 0
        state['best_score'] = float('inf')
        state["failed_iter"] = 0
        state["decrease_count"] = 0
        if bst is not None:
            if bst.attr('best_score') is not None:
                state['best_score'] = float(bst.attr('best_score'))
                state['best_iteration'] = int(bst.attr('best_iteration'))
                state['best_msg'] = bst.attr('best_msg')
            else:
                bst.set_attr(best_iteration=str(state['best_iteration']))
                bst.set_attr(best_score=str(state['best_score']))
        else:
            assert env.cvfolds is not None

        bst.set_param("learning_rate", start_lr)

    def callback(env):

        """internal function"""
        score = env.evaluation_result_list[-1][1]
        if len(state) == 0:
            init(env)
        best_score = state['best_score']
        best_iteration = state['best_iteration']

        if score < best_score:
            state['best_score'] = score
            state['best_iteration'] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(best_score=str(state['best_score']),
                                   best_iteration=str(state['best_iteration']))

        elif env.iteration - best_iteration >= rounds_function(state["decrease_count"]):
            if state["failed_iter"] < rounds_function(state["decrease_count"]):
                state["failed_iter"] += 1
            else:
                env.model.set_param("learning_rate", decrease_function(state["cur_lr"]))
                state["cur_lr"] = decrease_function(state["cur_lr"])
                print("lowered lr", state["cur_lr"])
                if state["cur_lr"] < min_lr:
                    print("Reached minimal LR, stopped. LR:", state["cur_lr"], "Score:", state['best_score'])
                    raise EarlyStopException(best_iteration)

                state["failed_iter"] = 0
                state["decrease_count"] += 1

    return callback


def bold_driver(start_lr, min_lr, boldness, timidness, relax, relax_k):
    state = {}


    def init(env):
        bst = env.model

        if len(env.evaluation_result_list) == 0:
            raise ValueError('For LR-based early stopping you need at least one set in evals.')

        state["cur_lr"] = start_lr
        state['best_iteration'] = 0
        state['prev_score'] = float('inf')
        state["relax"] = relax
        state["relaxation_rounds"] = 0
        bst.set_param("learning_rate", start_lr)

    def callback(env):
        """internal function"""
        score = env.evaluation_result_list[-1][1]
        if len(state) == 0:
            init(env)
        prev_score = state['prev_score']
        best_iteration = state['best_iteration']


        if score < prev_score:
            state['prev_score'] = score
            state['best_iteration'] = env.iteration
            state["cur_lr"] *= boldness
            env.model.set_param("learning_rate", state["cur_lr"])
        else:

            state['prev_score'] = score
            if state["relaxation_rounds"] < 0:
                print("Reduced LR from", state["cur_lr"], "to", state["cur_lr"] * timidness)
                state["relaxation_rounds"] = state["relax"] * relax_k
                state["relax"] = state["relaxation_rounds"]
                state["cur_lr"] *= timidness

            else:
                state["relaxation_rounds"] -= 1
            env.model.set_param("learning_rate", state["cur_lr"])
            if state["cur_lr"] < min_lr:
                print("Reached minimal LR, stopped. LR:", state["cur_lr"], "Last Score:", state['prev_score'])
                raise EarlyStopException(best_iteration)

    return callback


def mc_clain(start_lr, target_lr):
    state = {}


    def init(env):
        bst = env.model

        if len(env.evaluation_result_list) == 0:
            raise ValueError('For LR-based early stopping you need at least one set in evals.')

        state["cur_lr"] = start_lr
        state['best_iteration'] = 0
        state['prev_score'] = float('inf')
        state["relaxation_rounds"] = 0
        bst.set_param("learning_rate", start_lr)


    def callback(env):
        """internal function"""
        score = env.evaluation_result_list[-1][1]
        if len(state) == 0:
            init(env)
        prev_score = state['prev_score']
        best_iteration = state['best_iteration']

        prev_lr = state["cur_lr"]
        lr = prev_lr / (prev_lr + 1 - target_lr)
        env.model.set_param("learning_rate", lr)
        state["cur_lr"] = lr



    return callback


def stc(start_lr, T):
    state = {}


    def init(env):
        bst = env.model

        if len(env.evaluation_result_list) == 0:
            raise ValueError('For LR-based early stopping you need at least one set in evals.')

        state["start_lr"] = start_lr
        state['best_iteration'] = 0
        bst.set_param("learning_rate", start_lr)


    def callback(env):
        """internal function"""
        if len(state) == 0:
            init(env)

        lr = state["start_lr"] / (1 + env.iteration / T)
        env.model.set_param("learning_rate", lr)


    return callback


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


class WithRunningMean(BaseLRMeta):
    def __new__(mcs, name, bases, dic):
        cls = super(WithRunningMean, mcs).__new__(mcs, name, bases, dic)

        def wrap_init(__init__):
            @wraps(__init__)
            def wrapped(self, *args, **kwargs):
                __init__(self, *args, **kwargs)
                self.rmean=None
                self.b = kwargs.pop('b', .9)
            return wrapped

        def wrap_call(__call__):
            @wraps(__call__)
            def wrapped(self, env):
                self.rmean = self.b1 * self.rmean + (1 - self.b1) * self.trace.values()[0][-1]
                __call__(self, env)
            return wrapped

        cls.__init__ = wrap_init(cls.__init__)
        cls.__call__ = wrap_call(cls.__call__)
        return cls


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
            print(lr, e, diff, coef_, coef, file=open('log.txt', 'a'))
            env.model.set_param("learning_rate", lr)

