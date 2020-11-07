#################################
# Your name: Hadar Tal
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    
    """Assignment 2 skeleton.
    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """
    
    figure_number = 0
    
    def new_figure(self):
        figure_number +=1
        plt.figure(figure_number)
       
        
    def predict(self, intervals, x):
        """
        Helper Method for get the prediction of a point by given intervals.
        return - 1 if the intervals contains the point x, o.w 0
        """
        for interval in intervals:
            if x >= interval[0] and x <= interval[1]:
                return 1
        return 0    


    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.
        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        # x is distributed uniformly on the interval [0,1]
        mat = np.ndarray(shape=(m,2))
        y_classes = np.arange(0,2)
        min_1 = 0 
        max_1 = 0.2
        min_2 = 0.4
        max_2 = 0.6
        min_3 = 0.8
        max_3 = 1
        dist_a = [0.2,0.8]
        dist_b = [0.9,0.1]
        for row in mat:
            x = np.random.random()
            if (x >= min_1 and x <= max_1) or (x >= min_2 and x <= max_2) or (x >= min_3 and x <= max_3):
                y = np.random.choice(y_classes, p=dist_a)
            else:
                y = np.random.choice(y_classes, p=dist_b)
            row[0] = x
            row[1] = y
        mat = mat[mat[:, 0].argsort()]
        return mat
        
    
    def create_sample_and_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: np.ndarray of shape (m,2) and the intervals.
        """
        mat = self.sample_from_D(m)
        interval_arr = intervals.find_best_interval(mat[:,0],mat[:,1],k)[0]
        return (mat, interval_arr)
        

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        (mat, intervals) = self.create_sample_and_intervals(m, k)
        plt.plot(mat[:,0],mat[:,1], '.', color="black")
        plt.ylim(-0.1, 1.1)
        plt.xlim(0,1)
        for x in (0.2,0.4,0.6,0.8):
            plt.axvline(x=x)
        for interval in intervals:
           plt.fill_between(interval,1.1, -0.1, color="red", alpha=0.2)


    def calculate_empirical_error(self, mat, intervals):
        """
        Calculates the empirical error.
        Input: np.ndarray of shape (m,2) and the intervals.
        Returns: the empirical error.
        """
        error_rate = 0
        length = len(mat)
        for row in mat:
            prediction = self.predict(intervals,row[0])
            if(prediction != row[1]):
                error_rate += 1
        return error_rate/length
    
    
    def calculate_true_error(self, intervals):
        """
        Input -intervals 
        returns - the true error of the intervals
        Because the interval is in size 1 , the total length that is not
        classified correcrly equals the true error
        """
        arr = []
        vals = [(0,0),(0.2,0),(0.4,0),(0.6,0),(0.8,0),(1,0)]
        inter_vals = []
        for interval in intervals:
            inter_vals.append((interval[0],"start"))
            inter_vals.append((interval[1],"end"))
        # need to check if to filter possible 2 same x values
        print(inter_vals)
        i = j = 0 
        while(i != len(vals) and j != len(inter_vals)):
            if(vals[i][0] < inter_vals[j][0]):
                arr.append(vals[i])
                i+=1
            else:
                arr.append(inter_vals[j])
                j+=1
        if i == len(vals):
            while(j != len(inter_vals)):
                arr.append(inter_vals[j])
                j+=1
        else: 
            while(i != len(vals)):
                arr.append(vals[i])
                i+=1
        new_arr = []
        i = 0
        while i != len(arr)-1:
            new_elem = (arr[i],arr[i+1])
            new_arr.append(new_elem)
            i+=1
        print(new_arr)
        return 0
            
        

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        for m in range(m_first,m_last+1):
            sum_of_empirical_errors = 0
            sum_of_true_errors = 0
            for t in range(0,T):
                (mat, intervals) =  self.create_sample_and_intervals(m, k)
                empirical_error = self.calculate_empirical_error(mat,intervals)
                sum_of_empirical_errors += empirical_error
                true_error = self.calculate_true_error(intervals)
                sum_of_true_errors += true_error
            avg_empirical_error = sum_of_empirical_errors/T
            avg_true_error = sum_of_true_errors/T
            print("m : " + str(m) + 
                  " , avg_empirical_error :" + str(avg_empirical_error) + 
                  " , avg_true_error :" + str(avg_true_error) + " .")
            
        
        
        

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # TODO: Implement the loop
        pass

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        # TODO: Implement the loop
        pass

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me
        pass

    #################################
    # Place for additional methods


    #################################


if __name__ == '__main__':
    
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3) 
    ass.experiment_m_range_erm(10, 10, 5, 3, 1)
    """
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)
    """

