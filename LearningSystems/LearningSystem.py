class LearningSystem(object):
    """
    The abstract base class for LearningSystems. The following functions must be implemented:
        get_path, __str__, fit, score
    """
    def get_path(self):
        raise NotImplementedError("This LearningSystem does not have a get_path function implemented")

    def __str__(self):
        raise NotImplementedError("This LearningSystem does not have a string function implemented")

    def fit(self, X, y):
        raise NotImplementedError("This LearningSystem does not have a fit function implemented")

    def score(X, y):
        raise NotImplementedError("This LearningSystem does not have a score function implemented")