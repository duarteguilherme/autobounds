class Program:
    """ This class
    will state a optimization program
    to be translated later to any 
    language of choice, pyscipopt (pip, pyscipopt(cip),
    pyomo, among others
    A program requires first parameters, 
    an objective function,
    and constraints
    """
    def __init__(self):
        self.parameters = [ ]
        self.estimand = tuple()
        self.constraints = [ tuple() ]
    
    def to_pyomo(self):
        pass
    
    def to_pip(self):
        pass
    
    def to_cip(self):
        pass

