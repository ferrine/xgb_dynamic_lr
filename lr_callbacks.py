from xgboost.core import EarlyStopException
from abc import ABCMeta


class BaseLR(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.inited = False

    def __init(self, env):
        self._init(env)
        self.inited = True

    def __call__(self, env):
        if not self.inited:
            self.__init(env)
        self.call(env)

    def _init(self, env):
        """If it is first call this method will be called
        override it for your needs, by default does nothing
        """

    def call(self, env):
        """Write callback logic here, be sure that _init is already called
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
