import time
import winsound

from Models import *

DATA_DIR = "C:\\Users\\Michał Pilichowski\\PycharmProjects\\INL_projekt\\data\\"
TRAIN_DATASET = f'{DATA_DIR}\\kpwr-time_TRAIN.txt'
TEST_DATASET = f'{DATA_DIR}\\kpwr-time_TEST.txt'

if __name__ == '__main__':
    print('main() - start.')
    start_time = time.time()

    try:
        run_crf(TRAIN_DATASET, TEST_DATASET, epoch_count=1000, display_diagrams=True, mode='all_but_word_morf')

        # run_nn(TRAIN_DATASET, TEST_DATASET, model_name='dense', epoch_count=25, display_diagrams=True)

        # run_logistic_regression(TRAIN_DATASET, TEST_DATASET, max_iteration_count=2_000, display_diagrams=True)

        # run_nn(TRAIN_DATASET, TEST_DATASET, model_name='lstm', epoch_count=25, display_diagrams=True)
    finally:
        print(f"Execution time: {round(time.time() - start_time, 5)} seconds")
        print(f"Execution time: {round((time.time() - start_time) / 60, 2)} minutes")
        winsound.Beep(440, 500)
        print('main() - end.')
