from sklearn.base import BaseEstimator, ClassifierMixin
import subprocess
import numpy as np
import os


class BayesPointMachine(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, iterations=30, batches=1, compute_evidence=False,
                 bin_dir="./bin",
                 train_file="train.txt",
                 test_file="test.txt",
                 prediction_file="predictions.txt",
                 model_file="trained-binary-bpm.bin",
                 multiclass=False):
        """
        Called when initializing the classifier
        """
        self.bin_dir = bin_dir
        self.iterations = iterations
        self.batches = batches
        self.compute_evidence = compute_evidence
        self.train_file = train_file
        self.test_file = test_file
        self.prediction_file = prediction_file
        self.model_file = model_file
        self.multiclass = multiclass
        self.trained = False
        self.model_name = "BinaryBayesPointMachine" if not self.multiclass else "MulticlassBayesPointMachine"
        self._classes = dict()
        # self._classes = set()

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.
        """

        if len(y.shape) > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)

        self._classes = dict((c, y_c) for y_c, c in enumerate(set(y)))
        # self._classes = set(y)

        # print('Data shape: ' + str(X.shape))
        print('Classes: ' + str(self._classes))
        if len(self._classes) > 2 and not self.multiclass:
            raise ValueError("self.multiclass is False but we got %d classes" % len(self._classes))

        # First create the input file for Infer.NET
        self._create_input_file(self.train_file, X, y)

        # Then call the command line runner
        cmd = ["mono", os.path.join(self.bin_dir, "Learner.exe"), "Classifier", self.model_name, "Train",
               "--training-set", self.train_file,
               "--model", self.model_file]
        self._execute(cmd)

        self.trained = True

        return self

    # def decision_function(self, X):
    #     """
    #     Predict confidence scores for samples.
    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix}, shape = (n_samples, n_features)
    #         Samples.
    #     Returns
    #     -------
    #     array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
    #         Confidence scores per (sample, class) combination. In the binary
    #         case, confidence score for self.classes_[1] where >0 means this
    #         class would be predicted.
    #     """
    #     return self.predict_proba(X)

    def predict_proba(self, X, y=None):
        if not self.trained:
            raise RuntimeError("You must train classifer before predicting data!")

        # print('Data shape: ' + str(X.shape))

        # First create the input file for Infer.NET
        self._create_input_file(self.test_file, X)

        # Then call the command line runner
        cmd = ["mono", os.path.join(self.bin_dir, "Learner.exe"), "Classifier", self.model_name, "Predict",
               "--test-set", self.test_file,
               "--model", self.model_file,
               "--predictions", self.prediction_file]
        self._execute(cmd)

        # Now load the predictions back in
        preds = list(self._get_predictions())
        return np.array(preds)

    def predict(self, X, y=None):
        return [np.argmax(yhat) for yhat in self.predict_proba(X, y)]

    def _execute(self, cmd):
        p = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(p.stdout.readline, ''):
            print(line)
        retval = p.wait()

    def _create_input_file(self, filename, X, y=None):
        if y is None:
            self._create_input_file(filename, X, [0 for i in range(len(X))])
            return
        with open(filename, 'w') as f:
            for x_i, y_i in zip(X, y):
                if hasattr(x_i, '__iter__'):
                    f.write('%d ' % y_i + ' '.join('%d:%f' % (j, x_ij) for j, x_ij in enumerate(x_i)) + '\n')
                else:
                    f.write('%d 0:%d\n' % (y_i, x_i))

    def _get_predictions(self):
        with open(self.prediction_file) as f:
            # 1=0.49321002813527 0=0.50678997186473
            for line in f:
                preds = np.empty((len(self._classes),))
                for pred in line.split(' '):
                    y, prob = pred.split('=')
                    preds[self._classes[int(y)]] = prob
                    # preds[int(y)] = prob
                yield preds


def un_binarize(y):
    return np.argmax(y, axis=1)

if __name__ == "__main__":
    # Noddy test
    X_train = [i for i in range(0, 100, 5)]
    X_test = [i + 3 for i in range(-5, 95, 5)]

    bpm = BayesPointMachine()
    bpm.fit(np.array(X_test), y=np.array([i / 10 for i in range(20)]))

    print('predictions: ' + str(bpm.predict(np.array(X_test))))

    # Multi-class
    from sklearn.datasets import make_blobs
    from sklearn.cross_validation import train_test_split

    n_samples = 900
    centers = [(-5, -5), (0, 0), (5, 5)]
    X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                      centers=centers, shuffle=False, random_state=42)
    bpm = BayesPointMachine(multiclass=True)
    y[:n_samples / 3] = 0
    y[n_samples / 3:n_samples * 2 / 3] = 1
    y[n_samples * 2 / 3:] = 4
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

    bpm.fit(X_train, y_train)
    y_hat = bpm.predict(X_test)
    prob_pos_clf = bpm.predict_proba(X_test)

    print('\n'.join("true=%d, pred=%d" % (y_i, y_i_hat) for (y_i, y_i_hat) in zip(y_train, y_hat)))
    # clf_score = brier_score_loss(y_test, prob_pos_clf)
    # print("No calibration: %1.3f" % clf_score)
    print('Error rate = %.4f' % np.mean([0.0 if int(y_i) == int(y_i_hat) else 1.0 for (y_i, y_i_hat)
                                        in zip(y_train.reshape(-1).tolist(), y_hat)]))
