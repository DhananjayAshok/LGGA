class LearningSystem(object):
    """
    The abstract base class for LearningSystems. The following functions must be implemented:
         __str__, fit, score
    Must also have the self.path variable initialized
    """
    def set_path(self, path):
        self.path = path

    def get_path(self):
        try:
            return self.path
        except:
            raise NotImplementedError("Must have a self.path variable")

    def get_predicted_equation(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError("This LearningSystem does not have a string function implemented")

    def fit(self, X, y):
        raise NotImplementedError("This LearningSystem does not have a fit function implemented")

    def score(X, y):
        raise NotImplementedError("This LearningSystem does not have a score function implemented")