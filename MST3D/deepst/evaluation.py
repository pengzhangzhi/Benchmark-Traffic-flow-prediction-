from tensorflow.keras.metrics import (
    RootMeanSquaredError,
    MeanAbsolutePercentageError
)
import numpy as np

def evaluate(y_true, y_pred, mmn, rmse_factor=1):
    def denormalize(x, mmn):
        return x * (mmn._max - mmn._min) / 2.
    
    def inverse_transform(X, mmn):
        X = (X + 1.) / 2.
        X = 1. * X * (mmn._max - mmn._min) + mmn._min
        return X
    
    def split_flow(X):
        inflow = X[:,0]
        outflow = X[:,1]
        return inflow, outflow
    
    def rmse(y_true, y_pred):
        m_factor = rmse_factor
        m = RootMeanSquaredError()
        m.update_state(y_true, y_pred)
        return denormalize(m.result().numpy(), mmn) * m_factor
    
    def mape(y_true, y_pred):
        y_true = inverse_transform(y_true, mmn)
        y_pred = inverse_transform(y_pred, mmn)
        idx = y_true > 1e-6

        m = MeanAbsolutePercentageError()
        m.update_state(y_true[idx], y_pred[idx])
        return m.result().numpy()
        # return np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])) * 100
    def mae(y_true, y_pred):
        """calculate mae loss"""
        y_true = inverse_transform(y_true, mmn)
        y_pred = inverse_transform(y_pred, mmn)
        return np.mean(np.abs(y_true - y_pred))
        
    def ape(y_true, y_pred):
        y_true = inverse_transform(y_true, mmn)
        y_pred = inverse_transform(y_pred, mmn)
        idx = y_true > 1e-6
        return np.sum(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])) * 100


    y_true_in, y_true_out = split_flow(y_true)
    y_pred_in, y_pred_out = split_flow(y_pred)

    score = []

    score.append(rmse(y_true_in, y_pred_in))
    score.append(rmse(y_true_out, y_pred_out))
    score.append(rmse(y_true, y_pred))
    score.append(mape(y_true_in, y_pred_in))
    score.append(mape(y_true_out, y_pred_out))
    score.append(mape(y_true, y_pred))
    score.append(ape(y_true_in, y_pred_in))
    score.append(ape(y_true_out, y_pred_out))
    score.append(ape(y_true, y_pred))
    score.append(mae(y_true, y_pred))
    print(
        f'rmse_in: {score[0]}\n'
        f'rmse_out: {score[1]}\n'
        f'rmse_total: {score[2]}\n'
        f'mape_in: {score[3]}\n'
        f'mape_out: {score[4]}\n'
        f'mape_total: {score[5]}\n'
        f'ape_out: {score[6]}\n'
        f'ape_out: {score[7]}\n'
        f'ape_total: {score[8]}\n'
        f'mae: {score[9]}\n'
    )
    return score
