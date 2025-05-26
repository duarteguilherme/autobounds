# This class defines results from a query
# Before this class, query would be returned as lists
# 
def remove_one(part):
    """ To be used inside clean_query(lst)
    If the second part of a query is a list with more than 1 element 
    and contains '1', this element must be removed
    """
    if len(part) > 1:
        return [ k for k in part if k != '1' ]
    else:
        return part

def clean_query(lst):
    """ This method sorts and removes duplicated and zero parameters 
    """
    lst = [ (x[0], sort_and_return_list(x[1])) for x in lst.copy() ] 
    duplicated = [ x[1]  # Removing duplicated
            for n, x in enumerate(lst) 
            if x[1] not in [ i[1] for i in lst[:n] ] ]
    lst = [ (sum([ i[0] 
        for i in lst if x == i[1] ]), x)
            for x in duplicated ]
    lst = [ (x[0], remove_one(x[1]) ) for x in lst if x[0] != 0 ] 
    return lst


def sort_and_return_list(x):
    x.sort()
    return x

class Query():
    def __init__(self, event, cond = None):
        self._event = self.verify_list(event)
        self._cond = self.verify_list(cond)

    def verify_list(self, lst):
        if lst is None:
            return 
        if isinstance(lst, list):
            _lst = lst
        elif isinstance(lst, str):
            _lst = [(1, [lst])]
        elif isinstance(lst, float):
            _lst = [(1 * lst, ['1'])]
        elif isinstance(lst, int):
            _lst = [(1 * lst, ['1'])]
        elif 'Query.Query' in str(type(lst)):
            _lst = lst._lst
        else:
            raise Exception('Verify argument')
        return _lst

    def __getitem__(self, item):
        return self._event[item]
    
    def __eq__(self, query2):
        sub = self.__sub__(query2)._event
        if len(sub._event) == 0 and len(sub._cond) == 0:
            return True
        else:
            return False
    
    def __str__(self):
        return f"{self._event[:]}"
    
    def __repr__(self):
        return f"{self._event[:]}"
    
    def __add__(self, query2):
        lst = self._event + query2._event
        return Query(clean_query(lst))
    
    def __sub__(self, query2):
        lst = []
        for i in query2._event.copy():
            lst.append( (i[0] * -1, i[1]))
        lst = self._event + lst
        return Query(clean_query(lst))
    
    def __mul__(self, query2):
        cond = [ ]
        for i in query2._event.copy():
            for j in self._event.copy():
                cond.append((i[0] * j[0], i[1] + j[1]))
        return Query(clean_query(cond))
    
    def clean(self):
        self._event = clean_query(self._event)
#        cond = query2._lst.copy()
#        for i in enumerate(cond):
#            cond[i][0] *= -1
#        lst = self._lst + cond
#        return clear_query(Query(lst))

