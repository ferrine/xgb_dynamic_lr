from xgboost.core import EarlyStopException

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