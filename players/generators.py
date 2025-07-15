class Generators:
    """
    A container class for managing multiple Generator objects.

    Attributes:
        generator_list (list): A list of Generator instances.
    """

    def __init__(self, generator_list):
        self.generator_list = generator_list

    ### Some getter functions to return all customer info in a list ###
    def get_all_gen_names(self):
        return [i.get_gen_name() for i in self.generator_list]
    
    def get_all_gen_types(self):
        return [i.get_gen_type() for i in self.generator_list]
        
    def get_all_gen_profiles(self):
        return [i.get_gen_profile() for i in self.generator_list]
    
    def get_all_gen_lcoes(self):
        return [i.get_gen_lcoe() for i in self.generator_list]
    
    def get_all_gen_names(self):
        return [i.get_gen_name() for i in self.generator_list]
    
    # Getter to access length of production profile through this class
    def get_gen_profile_len(self):
        return len(self.get_all_gen_profiles()[0])
    
    # Returns total number of generators stored in this class
    def get_gen_numbers(self):
        return len(self.generator_list)
    
    # Computes and returns the aggregated generation profile
    def get_agg_gen_profile(self):
        res = self.generator_list[0].get_gen_profile().copy()
        for i in  self.generator_list[1:]:
            res += i.get_gen_profile()
        return res

    # Getter to access the i-th generator
    def get_ith_gen(self, i):
        return self.generator_list[i]

    # Getter to access the list of generators    
    def get_generator_list(self):
        return self.generator_list
    
    # Adder to append a new Generator instance to this container class
    def add_new_gen(self, new_gen):
        self.generator_list.append(new_gen)