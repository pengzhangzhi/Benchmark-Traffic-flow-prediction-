import numpy as np
import os
from copy import copy

from star.evaluation import evaluate

def multi_step_2D(model, x_test, y_test, mmn, len_closeness, step):
    # model = build_model(external_dim)
    dict_multi_score = {}
    nb_flow = 2
    y_pre = []
    y_test = copy(y_test)
    x_test_now = [copy(e) for e in x_test]

    # inference
    for i in range(1, step + 1):
        y_pre_inference = model.predict(x_test_now)  # 1
        # expand dims [timeslots, flow, height, width] --> [step, timeslots, flow, height, width]
        y_pre_expand_dims = np.expand_dims(y_pre_inference, axis=0)
        # append in all step
        y_pre.append(y_pre_expand_dims)

        x_test_noremove = x_test_now[0][1:]
        x_test_noremove = x_test_noremove.transpose((1, 0, 2, 3))
        x_test_noremove = x_test_noremove[len_closeness * nb_flow:]
        x_test_noremove = x_test_noremove.transpose((1, 0, 2, 3))

        x_test_remove = x_test_now[0].transpose((1, 0, 2, 3))
        x_test_remove = x_test_remove[:len_closeness * nb_flow]

        y_pre_remove = y_pre_inference.transpose((1, 0, 2, 3))

        if len_closeness > 1:
            for j in range(len_closeness * nb_flow, 3, -nb_flow):
                x_test_remove[j - 2:j] = x_test_remove[j - 4:j - 2]  #
            x_test_remove[0:2] = y_pre_remove
        else:
            x_test_remove[0:2] = y_pre_remove

        x_test_remove = x_test_remove.transpose((1, 0, 2, 3))
        x_test_remove = x_test_remove[:-1]

        x_test_next = np.concatenate((x_test_remove, x_test_noremove), axis=1)
        #
        # make training data
        x_test_makeData = []

        x_test_makeData.append(x_test_next)
        x_test_makeData.append(x_test[1][i:])

        x_test_now = x_test_makeData


    for i in range(len(y_pre)):
        print(f'Step {i+1}:')
        score = evaluate(y_test[i:], y_pre[i][0], mmn)
        dict_multi_score[i] = score

    return dict_multi_score
