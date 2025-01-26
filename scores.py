import numpy as np
from scipy.special import psi, polygamma


# Dirichlet Score Computation
def inv_psi(y, iters=5):
    cond = y >= -2.22
    x = cond * (np.exp(y) + 0.5) + (1 - cond) * -1 / (y - psi(1))
    for _ in range(iters):
        x = x - (psi(x) - y) / polygamma(1, x)
    return x

def fixed_point_dirichlet_mle(alpha_init, log_p_hat, max_iter=1000):
    alpha_new = alpha_old = alpha_init
    for _ in range(max_iter):
        alpha_new = inv_psi(psi(np.sum(alpha_old)) + log_p_hat)
        if np.sqrt(np.sum((alpha_old - alpha_new) ** 2)) < 1e-9:
            break
        alpha_old = alpha_new
    return alpha_new

def dirichlet_normality_score(alpha, p):
    return np.sum((alpha - 1) * np.log(p), axis=-1)

def compute_scores(model, x_test, transformer, x_train_task):
    predictions=[]
    scores = np.zeros((len(x_test),))
    for t_ind in range(transformer.n_transforms):
        print(f"Applying transformation index: {t_ind}")

        observed_dirichlet = model.predict(transformer.transform_batch(x_train_task, [t_ind] * len(x_train_task)), batch_size=128)
        observed_dirichlet = np.clip(observed_dirichlet, 1e-10, 1.0)
        log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)

        alpha_sum_approx = len(observed_dirichlet) * (len(observed_dirichlet[0]) - 1) * (-psi(1))
        alpha_sum_approx /= len(observed_dirichlet) * np.sum(observed_dirichlet * np.log(observed_dirichlet)) - np.sum(observed_dirichlet * np.sum(np.log(observed_dirichlet), axis=0))
        alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx

        mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
        print(f"MLE Alpha (t_ind={t_ind}):", mle_alpha_t)

        x_test_p = model.predict(
            transformer.transform_batch(x_test, [t_ind] * len(x_test)), 
            batch_size=128)
        x_test_p = np.clip(x_test_p, 1e-10, 1.0)
        # scores += dirichlet_normality_score(mle_alpha_t, x_test_p)
        t_scores = dirichlet_normality_score(mle_alpha_t, x_test_p)
        print(f"Scores for transformation {t_ind}:", scores[:5])  # Debug: first 5 scores
        scores += t_scores

        predictions.append(x_test_p)
    scores /= transformer.n_transforms
    return scores, predictions


# def entropy_normality_score(probabilities):
#     """
#     Compute entropy-based normality score (lower entropy is more normal).
#     Args:
#         probabilities: Model's predicted probabilities for each sample.
#     Returns:
#         entropy: Negative entropy, where higher value means more normal.
#     """
#     entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-10, 1.0)), axis=1)
#     return -entropy  # Negative entropy for normality score

def entropy_normality_score(probabilities):
    """
    Compute entropy-based normality score (lower entropy is more normal).
    Args:
        probabilities: Model's predicted probabilities for each sample.
    Returns:
        entropy: Normalized negative entropy for normality scoring.
    """
    # Ensure probabilities are normalized
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6), "Probabilities are not normalized!"

    # Compute entropy
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-10, 1.0)), axis=1)

    # Optionally return raw entropy or its negative
    return -entropy  # Keep negative entropy if desired


def compute_scores_with_entropy(model, x_test, transformer, x_train_task):
    predictions = []
    scores = np.zeros((len(x_test),))
    for t_ind in range(transformer.n_transforms):
        print(f"Applying transformation index: {t_ind}")

        # observed_dirichlet = model.predict(transformer.transform_batch(x_train_task, [t_ind] * len(x_train_task)), batch_size=128)
        # observed_dirichlet = np.clip(observed_dirichlet, 1e-10, 1.0)
        # log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)

        # alpha_sum_approx = len(observed_dirichlet) * (len(observed_dirichlet[0]) - 1) * (-psi(1))
        # alpha_sum_approx /= len(observed_dirichlet) * np.sum(observed_dirichlet * np.log(observed_dirichlet)) - np.sum(observed_dirichlet * np.sum(np.log(observed_dirichlet), axis=0))
        # alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx

        # mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
        # print(f"MLE Alpha (t_ind={t_ind}):", mle_alpha_t)

        x_test_p = model.predict(transformer.transform_batch(x_test, [t_ind] * len(x_test)), batch_size=128)
        x_test_p = np.clip(x_test_p, 1e-10, 1.0)

        # Using entropy as the normality score
        t_scores = entropy_normality_score(x_test_p)
        print(f"Scores for transformation {t_ind}:", scores[:5])  # Debug: first 5 scores
        scores += t_scores

        predictions.append(x_test_p)
    scores /= transformer.n_transforms
    return scores, predictions