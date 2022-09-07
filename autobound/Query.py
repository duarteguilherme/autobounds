# This class defines results from a query
# Before this class, query would be returned as lists
# 
def clean_query(query):
    """ This method sorts and removes duplicated and zero parameters 
    """
    lst = [ (x[0], sorted1(x[1])) for x in query._lst.copy() ] 
    duplicated = [ x[1]  # Removing duplicated
            for n, x in enumerate(lst) 
            if x[1] not in [ i[1] for i in lst[:n] ] ]
    lst = [ (sum([ i[0] 
        for i in lst if x == i[1] ]), x)
            for x in duplicated ]
    lst = [ (x[0], x[1]) for x in lst if x[0] != 0 ] 
    return lst


def sorted1(list1):
    list1.sort()
    return list1

class Query():
    def __init__(self, lst):
        self._lst = lst
    
    def __getitem__(self, item):
        return self._lst[item]
    
    def __str__(self):
        return f"{self._lst[:]}"
    
    def __repr__(self):
        return f"{self._lst[:]}"
    
    def __add__(self, query2):
        lst = self._lst + query2._lst
        return clean_query(Query(lst))
    
    def __sub__(self, query2):
        lst2 = []
        for i in query2._lst.copy():
            lst2.append( (i[0] * -1, i[1]))
        lst = self._lst + lst2
        return clean_query(Query(lst))
    
    def clean(self):
        self._lst = clean_query(self._lst)
    
    def __mul__(self, query2):
        pass
#        lst2 = query2._lst.copy()
#        for i in enumerate(lst2):
#            lst2[i][0] *= -1
#        lst = self._lst + lst2
#        return clear_query(Query(lst))

