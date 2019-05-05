### Metrics
import math

def sim_pearson(items1, items2):
    """ Возвращает коэффициент корреляции Пирсона между p1 и p2 """

    # Получить список предметов, оцененных обоими
    si={}
    for item in items1:
        if item in items2: si[item]=1

    # Найти число элементов
    n=len(si)

    # Если нет ни одной общей оценки, вернуть 0
    if n==0: return 0

    # Вычислить сумму всех предпочтений
    sum1=sum([items1[it] for it in si])
    sum2=sum([items2[it] for it in si])

    # Вычислить сумму квадратов
    sum1Sq=sum([(items1[it]*items1[it]) for it in si])
    sum2Sq=sum([(items2[it]*items2[it]) for it in si])

    # Вычислить сумму произведений
    pSum=sum([items1[it]*items2[it] for it in si])

    # Вычислить коэффициент Пирсона
    num=pSum-(sum1*sum2/n)
    den=math.sqrt((sum1Sq-(sum1*sum1)/n)*(sum2Sq-(sum2*sum2)/n))
    if den==0: return 0

    r=num/den
    return r

def sim_distance(items1, items2):
    """
    Возвращает оценку подобия p1 и p2 на основе эвклидого расстояния
    """

    # Получить список предметов, оцененных обоими
    si={}
    for item in items1 or []:
        if item in items2:
            si[item]=1

    # Если нет ни одной общей оценки, вернуть 0
    if len(si)==0: return 0

    # Сложить квадраты разностей
    sum_of_squares=sum([((items1[item] - items2[item])*(items1[item] - items2[item]))
                             for item in items1 if item in items2])

    return 1/(1+sum_of_squares)