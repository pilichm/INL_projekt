import numpy as np
import pandas as pd
import seaborn as sns
import sklearn_crfsuite
import tensorflow as tf
from keras import layers, regularizers
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics

DATASET_COL_NAMES = ['ORDER_ID', 'TOKEN_ID', 'ORTH', 'LEMMA', 'POS', 'CTAG', 'LABEL', '_']


def load_train_test_data(train_filepath, test_filepath):
    # Get train data.
    train_df = pd.read_csv(train_filepath, delimiter='\t', index_col=False)
    train_df.columns = DATASET_COL_NAMES

    # Get test data.
    test_df = pd.read_csv(test_filepath, delimiter='\t', index_col=False)
    test_df.columns = DATASET_COL_NAMES

    # Delete rows and columns without values.
    test_df = test_df.drop('_', axis=1)
    test_df = test_df.dropna(how='any')

    train_df = train_df.drop('_', axis=1)
    train_df = train_df.dropna(how='any')

    return train_df, test_df


def display_confusion_matrix(y_true, y_prd):
    ax = plt.subplot()
    cm = confusion_matrix(y_true, y_prd)
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(np.unique(y_true))
    ax.yaxis.set_ticklabels(np.unique(y_true))

    plt.show()


def run_logistic_regression(train_filepath, test_filepath, max_iteration_count=1000, display_diagrams=False):
    train_df, test_df = load_train_test_data(train_filepath, test_filepath)
    print(train_df.head())

    train_data_list = []
    train_labels = []

    test_data_list = []
    test_labels = []

    train_data_to_list = train_df['ORTH'].values.tolist()
    train_labels_to_list = train_df['LABEL'].values.tolist()

    test_data_to_list = test_df['ORTH'].values.tolist()
    test_labels_to_list = test_df['LABEL'].values.tolist()

    print('Train ds values count:')
    print(train_df['LABEL'].value_counts())

    # Train data set value counts after balancing.
    train_df = train_df.drop(train_df[train_df['LABEL'] == 'O'].sample(frac=.99).index)
    print('Train ds values count:')
    print(train_df['LABEL'].value_counts())

    print('Test ds values count:')
    print(test_df['LABEL'].value_counts())

    # Map names to numbers.
    label_dict = {}
    value_dict = {}
    for num, label in enumerate(set(train_labels_to_list)):
        label_dict[label] = num
        value_dict[num] = label

    # Transform input text data with Tokenizer.
    for word, label in zip(train_data_to_list, train_labels_to_list):
        train_data_list.append(word)
        train_labels.append(label_dict[label])

    for word, label in zip(test_data_to_list, test_labels_to_list):
        test_data_list.append(word)
        test_labels.append(label_dict[label])

    train_data = np.array(train_data_list)
    test_data = np.array(test_data_list)

    max_words = 1000
    max_len = 200

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_data)

    train_sequences = tokenizer.texts_to_sequences(train_data)
    train_words = pad_sequences(train_sequences, maxlen=max_len)

    test_sequences = tokenizer.texts_to_sequences(test_data)
    test_words = pad_sequences(test_sequences, maxlen=max_len)

    test_labels = np.array(test_labels)
    train_labels = np.array(train_labels)

    # Create model and make predictions.
    clf = LogisticRegression(solver='lbfgs', max_iter=max_iteration_count)
    clf.fit(train_words, train_labels)

    predicted_prob = clf.predict_proba(test_words)
    predictions = predicted_prob.argmax(axis=1)

    # Convert predictions to labels.
    predictions = [item for item in predictions.astype(int)]
    test_labels = [item for item in test_labels.astype(int)]

    for num, prd in enumerate(predictions):
        predictions[num] = value_dict[prd]

    for num, label in enumerate(test_labels):
        test_labels[num] = value_dict[label]

    # Calculate precision, recall, f1-score and support for all classes separately.
    print(classification_report(y_true=test_labels, y_pred=predictions))

    # Display confusion matrix for predicted values.
    if display_diagrams:
        display_confusion_matrix(test_labels, test_labels)


def run_nn(train_filepath, test_filepath, epoch_count=100, display_diagrams=False):
    train_df, test_df = load_train_test_data(train_filepath, test_filepath)

    print("First rows of train data:")
    print(train_df.head())

    print("First rows of test data:")
    print(test_df.head())

    train_data_list = []
    train_labels = []

    test_data_list = []
    test_labels = []

    train_data_to_list = train_df['ORTH'].values.tolist()
    train_labels_to_list = train_df['LABEL'].values.tolist()

    test_data_to_list = test_df['ORTH'].values.tolist()
    test_labels_to_list = test_df['LABEL'].values.tolist()

    print('Train ds values count:')
    print(train_df['LABEL'].value_counts())

    # Train data set value counts after balancing.
    train_df = train_df.drop(train_df[train_df['LABEL'] == 'O'].sample(frac=.99).index)
    print('Train ds values count:')
    print(train_df['LABEL'].value_counts())

    print('Test ds values count:')
    print(test_df['LABEL'].value_counts())

    # Map names to numbers.
    label_dict = {}
    value_dict = {}
    for num, label in enumerate(set(train_labels_to_list)):
        label_dict[label] = num
        value_dict[num] = label

    print(label_dict)

    for word, label in zip(train_data_to_list, train_labels_to_list):
        train_data_list.append(word)
        train_labels.append(label_dict[label])

    for word, label in zip(test_data_to_list, test_labels_to_list):
        test_data_list.append(word)
        test_labels.append(label_dict[label])

    train_data = np.array(train_data_list)
    test_data = np.array(test_data_list)

    max_words = 1000
    max_len = 200
    num_of_labels = len(set(train_labels))

    train_labels = tf.keras.utils.to_categorical(train_labels, num_of_labels, dtype="float32")
    test_labels = tf.keras.utils.to_categorical(test_labels, num_of_labels, dtype="float32")

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_data)

    train_sequences = tokenizer.texts_to_sequences(train_data)
    train_words = pad_sequences(train_sequences, maxlen=max_len)

    test_sequences = tokenizer.texts_to_sequences(test_data)
    test_words = pad_sequences(test_sequences, maxlen=max_len)

    test_labels = np.array(test_labels)
    train_labels = np.array(train_labels)

    x_train, x_test, y_train, y_test = train_test_split(train_words, train_labels, random_state=0, test_size=0.3)

    # Create model.
    model = Sequential()
    model.add(layers.Embedding(max_words, 40, input_length=max_len))
    model.add(layers.Conv1D(20, 6, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),
                            bias_regularizer=regularizers.l2(2e-5)))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Bidirectional(layers.LSTM(20, dropout=0.05)))
    model.add(layers.Dense(num_of_labels, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.Precision(name='precision')])

    # Display model description.
    print(model.summary())

    model.fit(x_train,
              y_train,
              epochs=epoch_count,
              validation_data=(x_test, y_test))

    test_loss, test_acc, *rest = model.evaluate(x_test, y_test, verbose=2)
    print('Model test accuracy: ', test_acc)

    predictions = model.predict(test_words)
    predictions = np.around(predictions, decimals=0).argmax(axis=1)

    validation_loss, validation_acc, *rest = model.evaluate(test_words, test_labels, verbose=2)
    print('Model validation accuracy: ', validation_acc)

    test_labels = np.around(test_labels, decimals=0).argmax(axis=1)
    print(classification_report(y_true=test_labels, y_pred=predictions))

    predictions = [item for item in predictions.astype(int)]
    test_labels = [item for item in test_labels.astype(int)]

    for index in range(len(test_labels)):
        test_labels[index] = value_dict[test_labels[index]]
        predictions[index] = value_dict[predictions[index]]

    # Display confusion matrix for predicted values.
    if display_diagrams:
        display_confusion_matrix(test_labels, predictions)


def run_crf(train_filepath, test_filepath, epoch_count=100, display_diagrams=False, mode='all'):
    train_df, test_df = load_train_test_data(train_filepath, test_filepath)

    print("First rows of train data:")
    print(train_df.head())

    print("First rows of test data:")
    print(test_df.head())

    def convert_to_features_and_labels(data, in_mode='all'):
        sentence = []
        sentence_labels = []
        x = []
        y = []

        for index, row in data.iterrows():
            if row['TOKEN_ID'] == 0:
                if len(sentence) > 0:
                    x.append(sentence)
                    y.append(sentence_labels)
                sentence = []
                sentence_labels = []

            if in_mode in ['word_param_prev_next', 'all_but_word_morf', 'all']:
                if len(sentence) == 0:
                    prev_word = "BOS"
                    second_prev_word = "BOS"
                elif len(sentence) == 1:
                    second_prev_word = "BOS"
                    prev_word = sentence[len(sentence) - 1]['word.lower()']
                else:
                    second_prev_word = sentence[len(sentence) - 1]['word.prev_word']
                    prev_word = sentence[len(sentence) - 1]['word.lower()']
                    sentence[len(sentence) - 1]['word.next_word'] = row['ORTH'].lower()
                    sentence[len(sentence) - 2]['word.second_next_word'] = row['ORTH'].lower()

            if in_mode == 'word_only':
                features = {
                    'bias': 1,
                    'word.lower()': row['ORTH'].lower()
                }
            elif in_mode == 'word_and_params':
                features = {
                    'bias': 1,
                    'word.lower()': row['ORTH'].lower(),
                    'world.len()': len(row['ORTH']),
                    'word.isdigit()': row['ORTH'].isdigit(),
                    'word.istitle()': row['ORTH'].istitle()
                }
            elif in_mode == 'word_param_prev_next':
                features = {
                    'bias': 1,
                    'word.lower()': row['ORTH'].lower(),
                    'world.len()': len(row['ORTH']),
                    'word.isdigit()': row['ORTH'].isdigit(),
                    'word.istitle()': row['ORTH'].istitle(),
                    'word.prev_word': prev_word,
                    'word.next_word': "EOS"
                }
            elif in_mode == 'all_but_word_morf':
                features = {
                    'bias': 1,
                    'word.lower()': row['ORTH'].lower(),
                    'world.len()': len(row['ORTH']),
                    'word.isdigit()': row['ORTH'].isdigit(),
                    'word.istitle()': row['ORTH'].istitle(),
                    'word.prev_word': prev_word,
                    'word.second_prev_word': second_prev_word,
                    'word.next_word': "EOS",
                    'word.second_next_word': "EOS"
                }
            elif in_mode == 'all':
                features = {
                    'bias': 1,
                    'word.lower()': row['ORTH'].lower(),
                    'world.len()': len(row['ORTH']),
                    'word.isdigit()': row['ORTH'].isdigit(),
                    'word.istitle()': row['ORTH'].istitle(),
                    'word.prev_word': prev_word,
                    'word.second_prev_word': second_prev_word,
                    'word.next_word': "EOS",
                    'word.second_next_word': "EOS",
                    'word.pos': row['POS'].lower(),
                    'word.ctag': row['CTAG'].lower()
                }
            else:
                print('Invalid mode!')
                return

            sentence.append(features)
            sentence_labels.append(row['LABEL'])

        return x, y

    x_train, y_train = convert_to_features_and_labels(train_df, in_mode=mode)
    x_test, y_test = convert_to_features_and_labels(test_df, in_mode=mode)

    # Create model.
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.2,
        c2=0.2,
        max_iterations=epoch_count,
        all_possible_transitions=True
    )

    crf.fit(x_train, y_train)

    # Remove most common class.
    labels = list(crf.classes_)
    labels.remove('O')

    # Make predictions
    y_prd = crf.predict(x_test)

    # Calculate overall f1-score for whole test dataset.
    f1_score = metrics.flat_f1_score(y_test, y_prd, average='weighted', labels=labels)
    print(f'Overall f1 score for whole dataset is {f1_score}')

    # Calculate precision, recall, f1-score and support for all classes separately.
    flat_y_test = [item for sublist in y_test for item in sublist]
    flat_y_prd = [item for sublist in y_prd for item in sublist]

    print(classification_report(y_true=flat_y_test, y_pred=flat_y_prd))

    # Display confusion matrix for test data.
    if display_diagrams:
        display_confusion_matrix(flat_y_test, flat_y_prd)
