from .base import BaseEstimator

class Ranginator(BaseEstimator):
    """

    """

    def predict(self, X, N=10):
        """

        """
        res = []
        for x in X:
            p1 = self.storage.get(x)
            scores=[(self.metric(p1, self.storage.get(p2)), p2)
                    for p2 in self.storage.generator() if p2 != p1]

            # Отсортировать список по убыванию оценок
            scores.sort()
            scores.reverse()
            res.append(scores[0:N])
        return res;