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
        self.constraints = [ tuple() ]
    
    def to_pyomo(self):
        pass
    
    def to_pip(self, filename, sense = 'max'):
        filep = open(filename, 'w')
        sense = 'MAXIMIZE' if sense == 'max' else 'MINIMIZE'
        filep.write(sense + '\n' + '  obj: objvar' + '\n')
        filep.write('\nSUBJECT TO\n')
        for i, c in enumerate(self.constraints):
            fipep.write(f'  a{i}: ' + ' + '.join([ 
                str(k[0]) * ' * '.join(k[1])   
                for k in c ]) + '\n') 
        filep.write('\nBOUNDS\n')
        for p in self.parameters:
            if p != 'objvar':
                filep.write(f'  0 <= {p} <= 1\n')
            else:
                filep.write(f'  -1 <= {p} <= 1\n')
        filep.write('\nEND')
        filep.close()
    
    def to_cip(self):
        pass

