# This class defines results from Q, which translates to strata events defined as probability over variables (P)
# See documentation for details
# 
def remove_one(part):
    """
    Removes the element `1` or `'1'` from the second part of an algebraic term 
    if it exists, as it is considered redundant. For example, given a term 
    like (1, ['Z0', 'X00', 1]), the result will be (1, ['Z0', 'X00']).
    
    Args:
        part (list): The second part of an algebraic term (a list of elements).

    Returns:
        list: The cleaned list with `1` or `'1'` removed, if present.
    """
    if len(part) > 1:
        return [k for k in part if str(k) != '1']
    else:
        return part

def remove_zero_scalars(lst):
    """Removes terms with a scalar of 0."""
    return [term for term in lst if term[0] != 0]

def remove_duplicates(lst):
    """Combines terms with the same multiplicative part by summing their scalars."""
    result = {}
    for scalar, terms in lst:
        key = tuple(terms)
        if key in result:
            result[key] += scalar  # Combine scalars
        else:
            result[key] = scalar
    return [(scalar, list(key)) for key, scalar in result.items()]

def order_internal_lists(lst):
    """Orders the internal lists in each term alphabetically."""
    return [(scalar, sorted(terms)) for scalar, terms in lst]

def clean_list(lst):
    """Cleans the list by removing zero scalars, combining duplicates, 
    ordering internal lists alphabetically, and handling '1'."""
    lst = order_internal_lists(lst)  # Order internal lists alphabetically
    lst = remove_duplicates(lst)  # Combine duplicate terms
    lst = remove_zero_scalars(lst)  # Remove terms with zero scalars
    lst = [(scalar, remove_one(terms)) for scalar, terms in lst]  # Handle '1'
    return lst

def add_list(lst, addlst):
    """Adds addlst to lst based on the algebraic structure."""
    if lst is None or len(lst) == 0:
        return clean_list(addlst)
    if addlst is None or len(addlst) == 0:
        return clean_list(lst)

    # Combine the two lists
    combined = lst + addlst

    # Clean the combined list (remove zeros, combine duplicates, and handle '1')
    return clean_list(combined)

def sub_list(lst, sublst):
    """Subtracts sublst from lst based on the algebraic structure."""
    if lst is None or len(lst) == 0:
        # Negate all scalars in sublst and clean
        return clean_list([(-scalar, terms) for scalar, terms in sublst])
    if sublst is None or len(sublst) == 0:
        return clean_list(lst)

    # Negate all scalars in sublst
    negated_sublst = [(-scalar, terms) for scalar, terms in sublst]

    # Combine lst with the negated sublst
    combined = lst + negated_sublst

    # Clean the combined list (remove zeros, combine duplicates, and handle '1')
    return clean_list(combined)

def mul_list(lst1, lst2):
    """
    Multiplies two lists of algebraic terms.
    
    Args:
        lst1 (list): A list of terms, where each term is a tuple (scalar, [multiplicative parts]).
        lst2 (list): A list of terms, where each term is a tuple (scalar, [multiplicative parts]).
    
    Returns:
        list: A cleaned list of terms resulting from the multiplication.
    """
    result = []
    for scalar1, parts1 in lst1:
        for scalar2, parts2 in lst2:
            # Multiply scalars and concatenate multiplicative parts
            new_scalar = scalar1 * scalar2
            new_parts = parts1 + parts2
            result.append((new_scalar, new_parts))
    
    # Clean the result to simplify it
    return clean_list(result)

def ensure_q_instance(obj):
    """
    Ensures that the given object is an instance of the Q class.
    
    Args:
        obj: The object to check and convert.
    
    Returns:
        Q: An instance of the Q class.
    
    Raises:
        TypeError: If the object cannot be converted to an instance of Q.
    """
    if not isinstance(obj, Q):
        try:
            obj = Q(obj)
        except Exception as e:
            raise TypeError("The provided object cannot be converted to an instance of Q.") from e
    return obj

def compare_lists(lst1, lst2):
    """
    Compares two lists of algebraic terms for equality after cleaning them.
    
    Args:
        lst1 (list): First list of terms.
        lst2 (list): Second list of terms.
    
    Returns:
        bool: True if the cleaned lists are equal, False otherwise.
    """
    # Clean both lists
    cleaned_lst1 = clean_list(lst1)
    cleaned_lst2 = clean_list(lst2)
    
    # Compare the cleaned lists
    result = sub_list(cleaned_lst1, cleaned_lst2)
    
    # Check if the result is empty (all terms cancel out)
    return all(abs(scalar) < 1e-9 for scalar, _ in result)


def get_str_q(qlst):
    """ Get a string with the list
    """
    if qlst is None:
        return ""
    if len(qlst) == 0:
        return ""
    _qlst = ""
    for x in qlst:
        if x[0] == 1:
            _qlst += '*'.join(x[1]) + ' + '
        elif x[0] == -1:
            _qlst += '-*' + '*'.join(x[1]) + ' + '
        else:
            _qlst += str(x[0]) + '*' + '*'.join(x[1]) + ' + '
    if _qlst.endswith(' + '):
        _qlst = _qlst[:-3]
    return _qlst


class Q():
    def __init__(self, event, cond = None):
        self._event = clean_list(self.verify_list(event))
        self._cond = cond
        if cond is not None:
            self._cond = clean_list(self.verify_list(cond))
            if len(self._cond) == 0 or self._cond[0][0] == 0:
                raise Exception("Condition cannot have probability zero")

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
        elif 'Q.Q' in str(type(lst)):
            _lst = lst._event
        else:
            raise TypeError('Q() argument must be an int, float, or str')
        return _lst

#    def __getitem__(self, item):
#        return self._event[item]
    
    def __eq__(self, q2):
        """ Check if two qs are equal
        """
        ensure_q_instance(q2)
        if self._cond is None:
            if q2._cond is None:
                return compare_lists(self._event, q2._event)
            else:
                return False
        else:
            if q2._cond is None:
                return False
            return compare_lists(self._event, q2._event) and compare_lists(self._cond, q2._cond)
    
    def __str__(self):
        if self._cond is None:
            return f"Event: {get_str_q(self._event)}"
        return f"Event: {get_str_q(self._event)} \nCondition: {get_str_q(self._cond)}"
    
    def __repr__(self):
        if self._cond is None:
            return f"Event: {get_str_q(self._event)}"
        return f"Event: {get_str_q(self._event)} \nCondition: {get_str_q(self._cond)}"
        # if self._cond is None:
        #     return f"Event: {self._event[:]}"
        # return f"Event: {self._event[:]} \n Condition: {self._cond[:]}"
     
    def __add__(self, q2):
        """ Add two queries together
        """
        q2 = ensure_q_instance(q2)
        if self._cond is None:
            if q2._cond is None:
                return Q(add_list(self._event, q2._event))
            else:
                return Q(add_list(mul_list(self._event, q2._cond), q2._event))
        else:
            if q2._cond is None:
                return Q(add_list(self._event, mul_list(q2._event, self._cond)))
            else:
                if compare_lists(self._cond, q2._cond):
                    return Q(add_list(self._event, q2._event), self._cond)
                return Q(add_list(mul_list(self._event, q2._cond), mul_list(q2._event, self._cond)))
    
    def __sub__(self, q2):
        """ Subtract q from self
        """
        q2 = ensure_q_instance(q2)
        if self._cond is None:
            if q2._cond is None:
                return Q(sub_list(self._event, q2._event))
            else:
                return Q(sub_list(mul_list(self._event, q2._cond), q2._event))
        else:
            if q2._cond is None:
                return Q(sub_list(self._event, mul_list(q2._event, self._cond)))
            else:
                if compare_lists(self._cond, q2._cond):
                    return Q(sub_list(self._event, q2._event), self._cond)
                return Q(sub_list(mul_list(self._event, q2._cond), mul_list(q2._event, self._cond)))
            
    def __mul__(self, q2):
        """ Multiply two queries together
        """
        q2 = ensure_q_instance(q2)
        if self._cond is None:
            if q2._cond is None:
                return Q(mul_list(self._event, q2._event))
            else:
                return Q(mul_list(self._event, q2._event), q2._cond)
        else:
            if q2._cond is None:
                return Q(mul_list(self._event, q2._event), self._cond)
            else:
                return Q(mul_list(self._event, q2._event), mul_list(self._cond, q2._cond))
    
    def clean(self):
        self._event = clean_list(self._event)
        if self._cond is not None:
            self._cond = clean_list(self._cond)


Query = Q    

