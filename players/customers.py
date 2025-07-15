class Customers:
    """
    A container class for managing multiple Customer objects.

    Attributes:
        customer_list (list): A list of Customer instances.
    """

    def __init__(self, customer_list):
        self.customer_list = customer_list

    ### Some getter functions to return all customer info in a list ###
    def get_all_cus_names(self):
        return [i.get_cus_name() for i in self.customer_list]
    
    def get_all_cus_types(self):
        return [i.get_cus_type() for i in self.customer_list]
        
    def get_all_cus_profiles(self):
        return [i.get_cus_profile() for i in self.customer_list]
    
    # Getter to access length of consumption profile through this class
    def get_cus_profile_len(self):
        return len(self.get_all_cus_profiles()[0])
   
   # Returns total number of customers stored in this class
    def get_cus_numbers(self):
        return len(self.customer_list)
    
    # Computes and returns the aggregated consumption profile
    def get_agg_cus_profile(self):
        res = self.customer_list[0].get_cus_profile().copy()
        for i in  self.customer_list[1:]:
            res += i.get_cus_profile()
        return res

    # Getter to access the i-th customer
    def get_ith_cus(self, i):
        return self.customer_list[i]
    
    # Getter to access the list of customers
    def get_customer_list(self):
        return self.customer_list