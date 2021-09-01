import numpy as np
import pandas as pd
from itertools import product
import random
from scipy import stats
from matplotlib import pyplot as plt


class Node:
    
    def __init__(self, entropy, num_samples, num_samples_per_class, pred_y):
        self.entropy = entropy
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.pred_y = pred_y
        self.feature_index = 0
        self.inf_gain = 0
        self.split_conditions = []
        self.children = None


class Adaboost:

    def __init__(self):
        self.num_classifiers = -1
        self.split_list = []
        self.hypothesis_index = []
        self.weighted_error = []
        self.alphas = []
        self.data_weights = []


def get_weighted_error(estimated, actual, weights):
    if (len(weights) != len(estimated)):
        print("Size Mis-Match")
    dataset_size = len(estimated)
    bool_list = [1 if estimated[i] != actual[i] else 0 for i in range(0, dataset_size)]
    error = np.sum(weights * bool_list)
    return error


def get_predictions(root, X):
    dataset_size = X.shape[0]
    predictions = list()
    for k in range(0, dataset_size):
        temp_node = root
        while temp_node.children != None:
            feature_idx = temp_node.feature_index
            split_variables = temp_node.split_conditions
            split_cond = X.loc[k, feature_idx]
            idx = split_variables.index(split_cond)
            temp_node = temp_node.children[idx]
        predictions.append(temp_node.pred_y)
    return np.array(predictions)


def entropy_cond(feature_vector, Y):
    unique_features = list(set(feature_vector))
    label_space = list(set(Y))
    prior_probs = dict()
    cond_probs = dict()
    cond_entropy = 0
    for feature in unique_features:
        prior_probs[feature] = len(np.where(feature_vector == feature)[0]) / len(feature_vector)
        y_wrt_feature = Y[feature_vector.index[np.where(feature_vector == feature)[0]]]
        temp = dict()
        for label in label_space:
            if len(y_wrt_feature) != 0:
                temp[label] = len(np.where(y_wrt_feature == label)[0]) / len(y_wrt_feature)
                cond_entropy = cond_entropy + prior_probs[feature] * np.log2(temp[label] ** temp[label])
            else:
                temp[label] = 0
        cond_probs[feature] = temp
    cond_entropy = -1 * cond_entropy
    return cond_entropy


def best_split(X, Y):
    cond_entropy = dict()
    for col in X.columns:
        cond_entropy[col] = entropy_cond(X[col], Y)
    idx = min(cond_entropy, key=lambda k: cond_entropy[k])
    min_entropy = min(cond_entropy.values())
    return idx, min_entropy


def calc_entropy(Y):
    label_space = list(set(Y))
    entropy = 0
    for label in label_space:
        no_wrt_label = len(np.where(Y == label)[0])
        prob = no_wrt_label / len(Y)
    return entropy - np.log2(prob ** prob)


def tree_growth(X, Y, parent_class, rem_splits):
    samples_per_class = {i: sum(Y == i) for i in set(Y)}
    if len(Y) != 0:
        max_class = max(samples_per_class, key=lambda k: samples_per_class[k])
    else:
        max_class = parent_class
    node = Node(entropy=calc_entropy(Y), num_samples=len(Y), num_samples_per_class=samples_per_class,
                pred_y=max_class)
    if (calc_entropy(Y) != 0) & (X.shape[1] != 0) & (len(Y) != 0) & (rem_splits > 0):
        idx, entropy_min = best_split(X, Y)
        node.inf_gain = node.entropy - entropy_min
        node.feature_index = idx
        unique_features = [0, 1]
        node.split_conditions = unique_features
        child_list = []
        for feature in unique_features:
            feature_indexes = X.index[np.where(X[idx] == feature)[0]]
            x_new = X.loc[feature_indexes, X.columns != idx]
            y_new = Y[feature_indexes]
            child_list.append(tree_growth(x_new, y_new, max_class, rem_splits - 1))
        node.children = child_list
    return node


def tree_split_growth(X, y, splits_idx, output_map, growth_direction):
    samples_per_class = {i: sum(y == i) for i in set(y)}
    if len(y) != 0:
        max_class = max(samples_per_class, key=lambda k: samples_per_class[k])
    root = Node(entropy=-1, num_samples=len(y), num_samples_per_class=samples_per_class,
                pred_y=max_class)
    root.feature_index = splits_idx[0]
    root.split_conditions = [0, 1]
    root.children = []
    if growth_direction == 'l':
        feature_indexes = X.index[np.where(X[splits_idx[0]] == 0)[0]]
        X_new = X.loc[feature_indexes]
        y_new = y[feature_indexes]
        samples_per_class = {i: sum(y_new == i) for i in set(y_new)}
        max_class = max(samples_per_class, key=lambda k: samples_per_class[k])
        child = Node(entropy=-1, num_samples=len(y_new), num_samples_per_class=samples_per_class,
                     pred_y=max_class)
        child.feature_index = splits_idx[1]
        child.split_conditions = [0, 1]
        child.children = []
        for j in range(0, 2):
            feature_indexes = X_new.index[np.where(X_new[splits_idx[1]] == j)[0]]
            X_new1 = X_new.loc[feature_indexes]
            y_new1 = y_new[feature_indexes]
            samples_per_class = {i: sum(y_new1 == i) for i in set(y_new1)}
            child1 = Node(entropy=-1, num_samples=len(y_new1), num_samples_per_class=samples_per_class,
                          pred_y=output_map[j])
            child1.feature_index = -1
            child1.split_conditions = []
            child.children.append(child1)
        root.children.append(child)
        feature_indexes = X.index[np.where(X[splits_idx[0]] == 1)[0]]
        y_new = y[feature_indexes]
        samples_per_class = {i: sum(y_new == i) for i in set(y_new)}
        child = Node(entropy=-1, num_samples=len(y_new), num_samples_per_class=samples_per_class,
                     pred_y=output_map[2])
        child.feature_index = -1
        child.split_conditions = []
        root.children.append(child)
    else:
        feature_indexes = X.index[np.where(X[splits_idx[0]] == 0)[0]]
        y_new = y[feature_indexes]
        samples_per_class = {i: sum(y_new == i) for i in set(y_new)}
        max_class = max(samples_per_class, key=lambda k: samples_per_class[k])
        child = Node(entropy=-1, num_samples=len(y_new), num_samples_per_class=samples_per_class,
                     pred_y=max_class)
        child.feature_index = -1
        child.split_conditions = []
        root.children.append(child)
        feature_indexes = X.index[np.where(X[splits_idx[0]] == 1)[0]]
        X_new = X.loc[feature_indexes]
        y_new = y[feature_indexes]
        samples_per_class = {i: sum(y_new == i) for i in set(y_new)}
        child = Node(entropy=-1, num_samples=len(y_new), num_samples_per_class=samples_per_class,
                     pred_y=output_map[0])
        child.feature_index = splits_idx[1]
        child.split_conditions = [0, 1]
        child.children = []
        for j in range(0, 2):
            feature_indexes = X_new.index[np.where(X_new[splits_idx[1]] == j)[0]]
            y_new1 = y_new[feature_indexes]
            samples_per_class = {i: sum(y_new1 == i) for i in set(y_new1)}
            child1 = Node(entropy=-1, num_samples=len(y_new1), num_samples_per_class=samples_per_class,
                          pred_y=output_map[j + 1])
            child1.feature_index = -1
            child1.split_conditions = []
            child.children.append(child1)
        root.children.append(child)
    return root


def two_split_decision_stump(X, y):
    decision_trees_list = []
    p = list(product([-1, 1], repeat=3))
    for i in X.columns:
        for j in X.columns:
            for k in ['l', 'r']:
                for l in p:
                    splits_idx = [i, j]
                    decision_trees_list.append(tree_split_growth(X, y, splits_idx, l, k))
    return decision_trees_list


def one_split_tree_growth(X, y, split_idx, output_map):
    samples_per_class = {i: sum(y == i) for i in set(y)}
    if len(y) != 0:
        max_class = max(samples_per_class, key=lambda k: samples_per_class[k])
    root = Node(entropy=-1, num_samples=len(y), num_samples_per_class=samples_per_class,
                pred_y=max_class)
    root.feature_index = split_idx
    root.split_conditions = [0, 1]
    root.children = []
    for j in range(0, 2):
        feature_indexes = X.index[np.where(X[split_idx] == j)[0]]
        X_new = X.loc[feature_indexes]
        y_new = y[feature_indexes]
        samples_per_class = {i: sum(y_new == i) for i in set(y_new)}
        child = Node(entropy=-1, num_samples=len(y_new), num_samples_per_class=samples_per_class,
                     pred_y=output_map[j])
        child.feature_index = -1
        child.split_conditions = []
        root.children.append(child)
    return root


def split_one_stump(X, y):
    decision_trees_list = []
    p = list(product([-1, 1], repeat=2))
    for i in X.columns:
        for l in p:
            splits_idx = i
            decision_trees_list.append(one_split_tree_growth(X, y, splits_idx, l))
    return decision_trees_list


def adaboost_algo(X, y, n, hypothesis_space):
    selected_split_list = []
    selected_hypothesis_error = []
    selected_hypothesis_alpha = []
    selected_hypothesis_index = []
    selected_hypothesis_weights = []
    m = len(y)
    w = pd.Series((1 / m) * np.ones(m))
    for i in range(0, n):
        weighted_error_list = []
        for node in hypothesis_space:
            preds = get_predictions(node, X)
            weighted_error_list.append(get_weighted_error(preds, y, w))
        idx = np.argmin(weighted_error_list)
        selected_split_list.append(hypothesis_space[idx])
        selected_hypothesis_error.append(weighted_error_list[idx])
        alpha = (1 / 2) * np.log((1 - weighted_error_list[idx]) / weighted_error_list[idx])
        selected_hypothesis_alpha.append(alpha)
        selected_hypothesis_index.append(idx)
        selected_hypothesis_weights.append(w)
        preds = get_predictions(hypothesis_space[idx], X)
        margin = np.exp(np.multiply(preds, y) * -1 * alpha)
        w_new = np.multiply(w, margin) / (
                2 * np.sqrt(weighted_error_list[idx] * (1 - weighted_error_list[idx])))
        w = w_new
    result = Adaboost()
    result.num_classifiers = n
    result.split_list = selected_split_list
    result.hypothesis_index = selected_hypothesis_index
    result.weighted_error = selected_hypothesis_error
    result.alphas = selected_hypothesis_alpha
    result.data_weights = selected_hypothesis_weights
    return result


def get_accuracy(X, y, clf, n):
    combined_pred = np.zeros(X.shape[0])
    for i in range(0, n):
        pred = get_predictions(clf.split_list[i], X)
        combined_pred = combined_pred + clf.alphas[i] * pred
    estimated = np.sign(combined_pred)
    estimated[np.where(combined_pred == 0)] = 0
    acc = 100 * (len(np.where(estimated == y)[0]) / len(y))
    return acc


def boosting_wrt_coordinate_descent(X, y, hypothesis_space):
    num_of_hypothesis = len(hypothesis_space)
    [m, n] = X.shape
    num_of_iterations = 0
    alphas = np.zeros(num_of_hypothesis)
    alphas = np.reshape(alphas, (1, num_of_hypothesis))
    preds_matrix = np.zeros((m, num_of_hypothesis))
    order = list(range(0, num_of_hypothesis))
    random.shuffle(order)
    for i in range(0, num_of_hypothesis):
        preds_matrix[:, i] = get_predictions(hypothesis_space[i], X)
    while True:
        difference_list = np.zeros(num_of_hypothesis)
        for i in order:
            eval_matrix = alphas * preds_matrix
            loss = np.sum(np.exp(
                -1 * np.reshape(np.array(y), (m, 1)) * np.reshape(np.sum(alphas * preds_matrix, axis=1), (m, 1))),
                          axis=0)
            eval_matrix[:, i] = 0
            sum_matrix = np.exp(-1 * np.reshape(np.array(y), (m, 1)) * np.reshape(np.sum(eval_matrix, axis=1), (m, 1)))
            binary = np.ones((m, 1))
            total = np.sum(binary * sum_matrix)
            binary[preds_matrix[:, i] != y] = 0
            temp = np.sum(binary * sum_matrix)
            alpha_new = 1 / 2 * np.log(temp / (total - temp))
            difference_list[i] = alpha_new - alphas[0, i]
            alphas[0, i] = alpha_new
            num_of_iterations += 1
            # print(num_of_iterations, loss)
        #print(num_of_iterations, loss, np.linalg.norm(difference_list))
        if np.linalg.norm(difference_list) < 1e-4:
            break
    return alphas


def bagging(X, y, n):
    no_of_samples = len(y)
    split_list = []
    for i in range(0, n):
        sample_indexes = [random.randrange(0, no_of_samples) for j in range(0, no_of_samples)]
        sample_data = X.loc[sample_indexes]
        sample_output = y[sample_indexes]
        split_list.append(tree_growth(sample_data, sample_output, 0, 1))
    return split_list


def get_bagging_accuracy(hypothesis, X, y):
    preds = np.zeros((len(y), len(hypothesis)))
    for i in range(0, len(hypothesis)):
        preds[:, i] = get_predictions(hypothesis[i], X)
    predictions = np.zeros(len(y))
    for i in range(0, len(y)):
        predictions[i] = stats.mode(preds[i, :]).mode
    acc = (len(np.where(predictions == y)[0]) / len(y)) * 100
    return acc


with open(r"heart_train.data", 'r') as csv_file:
    df = pd.read_csv(csv_file)
    train_output = df[df.columns[0]]
    train_data = df[df.columns[1:]]
    [m, n] = train_data.shape
    train_output = 2 * train_output - 1
    train_data.columns = list(range(0, n))

with open(r"heart_test.data", 'r') as csv_file:
    df = pd.read_csv(csv_file)
    test_output = df[df.columns[0]]
    test_data = df[df.columns[1:]]
    [m, n] = test_data.shape
    test_output = 2 * test_output - 1
    test_data.columns = list(range(0, n))


split_list = split_one_stump(train_data, train_output)
output = adaboost_algo(train_data, train_output, 20, split_list)
train_acc_list = []
test_acc_list = []
for i in range(1, 21):
    train_acc_list.append(get_accuracy(train_data, train_output, output, i))
    test_acc_list.append(get_accuracy(test_data, test_output, output, i))
plt.plot(list(range(1, 21)), train_acc_list)
plt.plot(list(range(1, 21)), test_acc_list)
plt.show()


split_list = two_split_decision_stump(train_data, train_output)
output = adaboost_algo(train_data, train_output, 10, split_list)
train_acc_list = []
test_acc_list = []
for i in range(1, 11):
    train_acc_list.append(get_accuracy(train_data, train_output, output, i))
    test_acc_list.append(get_accuracy(test_data, test_output, output, i))
plt.plot(list(range(1, 11)), train_acc_list)
plt.plot(list(range(1, 11)), test_acc_list)
plt.show()

split_list = split_one_stump(train_data, train_output)
alphas_wrt_descent = boosting_wrt_coordinate_descent(train_data, train_output, split_list)
output_wrt_descent = Adaboost()
output_wrt_descent.split_list = split_list
output_wrt_descent.hypothesis_index = list(range(0, len(split_list)))
output_wrt_descent.alphas = np.transpose(alphas_wrt_descent)
output_wrt_descent.num_classifiers = 88
print(get_accuracy(train_data, train_output, output_wrt_descent, 88))
print(get_accuracy(test_data, test_output, output_wrt_descent, 88))

split_list = bagging(train_data, train_output, 20)
print(get_bagging_accuracy(split_list, train_data, train_output))
print(get_bagging_accuracy(split_list, test_data, test_output))
