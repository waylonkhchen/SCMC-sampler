from parsing_constraints import *

class Constraint():
    """Constraints loaded from a file.
    
    Attributes
    ----------
    n_dim           : int,
        dimension of the domain
        
    example         : List,
        containing n_dim of float representing a feasible point in the domain
    
    exprs           : List, len = number of constraints
        containing code objects that represents the inequality, 
        each element can be evaluated and return a bool
        
    exprs_algebraic : List, len = number of constraints
        containing code objects that represents the algebraic expression before the inequality
        each element can be evaluated and return a float
        
    bounds: List
    
    
    Methods
    -------
    
    
    """


    def __init__(self, fname):
        """
        Construct a Constraint object from a constraints file

        :param fname: Name of the file to read the Constraint from (string)
        """
        with open(fname, "r") as f:
            lines = f.readlines()
            
        # Parse the dimension from the first line
        self.n_dim = int(lines[0])
        # Parse the example from the second line
        self.example = [float(x) for x in lines[1].split(" ")[0:self.n_dim]]

        # Run through the rest of the lines and compile the constraints
        self.exprs = []
#        self.exprs_algebraic = []
        self.functions = []
        self.exprs_string = ""
        self.exprs_list = []

        
        for i in range(2, len(lines)):

            if lines[i][0]  == "#":
                continue
            self.exprs.append(compile(lines[i], "<string>", "eval"))
#            self.exprs.append(compile(lines[i],'<string>', 'eval'))
#            self.exprs_algebraic.append(compile(lines[i].split(">=")[0].strip(),"<string>", "eval"))
            self.functions.append( 
                    (lambda i:
                      lambda x: eval(lines[i].split(">=")[0].strip())
                     )(i) #include (lambda i: <expression>) (i) into the context of the function
                    )
            self.exprs_string += ('\n'+lines[i])
            self.exprs_list.append(lines[i].strip())
                    
        self.bounds, self.exprs_list = self.constraints2bounds()
        
        #reassign self.functions to those that are non-trivial
        self.functions = []
        for line in self.exprs_list:                
            self.functions.append(
                    (lambda line:
                        lambda x: eval(line.split(">=")[0].strip())
                        )(line)
                                  )
        return


    def get_example(self):
        """Get the example feasible vector"""
        return self.example

    def get_ndim(self):
        """Get the dimension of the space on which the constraints are defined"""
        return self.n_dim
    
##somehow this doesn't work
    def get_functions(self):
        return self.functions
    
    def apply(self, x):
        """
        Apply the constraints to a vector, returning True only if all are satisfied

        :param x: list or array on which to evaluate the constraints
        """
        for expr in self.exprs:
            if not eval(expr):
                return False
        return True  
    def get_exprs_string(self):
        return self.exprs_string
    def get_exprs_list(self):
        return self.exprs_list
    
#new method for alloy test set
    def constraints2bounds(self):
        """
        """
        exprs_list = self.get_exprs_list()
        #separate constraints
        simpl, compl = filter_simple_constraints(exprs_list)
        
        bounds = []
        for expr  in simpl:
            parsed = parsing_simpl(expr)
            bounds.append(convert_parsed_to_bound(parsed))
        return bounds,compl


    
#    def evaluate_constraints(self, x):
#        """Evaluate each constraint at x"""
#        values=[]
#        for g_i in self.exprs_algebraic:
#            values.append(eval(g_i))
#        return values
