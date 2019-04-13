from .ranginator import Ranginator

class Collaborator(Ranginator):
    """Collaborative filtering estimator. Collaborative filtering is the process of filtering for information or patterns using techniques involving collaboration among multiple agents, viewpoints, data sources, etc. Applications of collaborative filtering typically involve very large data sets."""

    def predict(self, X, y=None, countX=10, minCoefX=0):
        res = []
        simX = Ranginator.predict(self, X, countX)

        for i, x in enumerate(simX):
            totals={}
            simSums={}
            myY = (self.storage.get(X[i]).keys()) or [] + (y or [])
            filteredSimX = [item for item in x if item[0] > minCoefX]

            for u in filteredSimX:
                stItems = self.storage.get(u[1])
                maxV = max(stItems.values())
                for item in stItems.keys():
                    # оценивать только фильмы, которые я еще не смотрел
                    if item not in myY:
                        # Коэффициент подобия * Оценка
                        totals.setdefault(item, 0)
                        totals[item] += (stItems[item]/maxV)*u[0]

                        # Сумма коэффициентов подобия
                        simSums.setdefault(item,0)
                        simSums[item]+=u[0]

            # Создать нормализованный список
            rankings=[(total/simSums[item],item) for item,total in totals.items( )]

            # Вернуть отсортированный список
            rankings.sort()
            rankings.reverse()

            res.append(rankings)

        return res









