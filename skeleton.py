#################################
# Your name: Hadar Tal
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals
import math

class Assignment2(object):
    
    """Assignment 2 skeleton.
    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """
        
        


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
        m_vec = []
        avg_empirical_error_vec = []
        avg_true_error_vec = []
        for m in range(m_first,m_last+1,step):
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
            m_vec.append(m)
            avg_empirical_error_vec.append(avg_empirical_error)
            avg_true_error_vec.append(avg_true_error)
        plt.figure(2)
        plt.plot(m_vec, avg_empirical_error_vec, color='red', marker='o', 
                 linestyle='dashed', linewidth=2, markersize=12)
        plt.plot(m_vec, avg_true_error_vec, color='green', marker='o', 
                 linewidth=2, markersize=12)        
        
        

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        mat = self.sample_from_D(m)
        k_vec = []
        empirical_error_vec = []
        true_error_vec = []
        for k in range(k_first, k_last+1,step):
            interval_arr = intervals.find_best_interval(mat[:,0],mat[:,1],k)[0]
            empirical_error = self.calculate_empirical_error(mat,interval_arr)
            true_error = self.calculate_true_error(interval_arr)

            print("k : " + str(k) + 
                  " , empirical_error :" + str(empirical_error) + 
                  " , true_error :" + str(true_error) + " .")
            k_vec.append(k)
            empirical_error_vec.append(empirical_error)
            true_error_vec.append(true_error)
        plt.figure(3)   
        plt.plot(k_vec, empirical_error_vec, color='red', marker='o', 
                 linestyle='dashed', linewidth=2, markersize=12)
        plt.plot(k_vec, true_error_vec, color='green', marker='o', 
                 linewidth=2, markersize=12)
        smallest_emp_err_k = k_vec[np.argsort(empirical_error_vec)[0]]
        print("smallest_emp_err_k: " + str(smallest_emp_err_k))
        return smallest_emp_err_k


    def calc_penalty(self, k, m, delta):
        return math.sqrt(8 * (math.log(4 / delta) + 2 * k * math.log(math.exp(1) * m / k)) / m)
        


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
        # according to q.3 - the VCdim of the subsets of
        # the real line formed by the union of k intervals is 2k
        mat = self.sample_from_D(m)
        k_vec = []
        empirical_error_vec = []
        penalty_vec = []
        true_error_vec = []
        for k in range(k_first, k_last+1,step):
            interval_arr = intervals.find_best_interval(mat[:,0],mat[:,1],k)[0]
            empirical_error = self.calculate_empirical_error(mat,interval_arr)
            true_error = self.calculate_true_error(interval_arr)
            penalty = self.calc_penalty(k,m,0.1)
            print("k : " + str(k) + 
                  " , empirical_error :" + str(empirical_error) + 
                  " , true_error :" + str(true_error) +
                  " , penalty :"+ str(penalty)+ " .")
            k_vec.append(k)
            empirical_error_vec.append(empirical_error)
            true_error_vec.append(true_error)
            penalty_vec.append(penalty)
        penalty_and_emp_error_vec = np.sum([empirical_error_vec,penalty_vec], axis=0)
        plt.figure(4)   
        plt.plot(k_vec, empirical_error_vec, color='red', marker='o', 
                 linestyle='dashed', linewidth=2, markersize=12)
        plt.plot(k_vec, true_error_vec, color='green', marker='o', 
                 linewidth=2, markersize=12)
        plt.plot(k_vec, penalty_vec, color='grey', marker='o', 
                 linewidth=2, markersize=12)
        plt.plot(k_vec, penalty_and_emp_error_vec, color='yellow', marker='o', 
                 linewidth=2, markersize=12)
        smallest_pen_emp_err_k = k_vec[np.argsort(penalty_and_emp_error_vec)[0]]
        print("smallest_pen_emp_err_k: " + str(smallest_pen_emp_err_k))  


    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        per = int(0.8*m)
        test_error_vec = np.zeros(10)
        for t in range(0,T):
            print(t)
            mat = self.sample_from_D(m)
            train = mat[:per]
            test = mat[per:]
            
            for k in range(1,11,1):
                print(k)
                interval_arr = intervals.find_best_interval(train[:,0],train[:,1],k)[0]
                print(interval_arr)
                test_error = self.calculate_empirical_error(test,interval_arr)
                print(test_error)
                test_error_vec[k-1] += test_error
        test_error_vec = np.divide(test_error_vec,T)
        plt.figure(5)   
        plt.plot(k_vec, test_error_vec, color='red', marker='o', 
                 linewidth=2, markersize=12)
        return 0



    #################################
    # Place for additional methods

    def predict(self, intervals, x):
        """
        Helper Method for get the prediction of a point by given intervals.
        return - 1 if the intervals contains the point x, o.w 0
        """
        for interval in intervals:
            if x >= interval[0] and x <= interval[1]:
                return 1
        return 0    


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
        emp_error = error_rate/length
        return emp_error
    
    
    def calculate_true_error(self, intervals):
        """
        Input -intervals 
        returns - the true error of the intervals
        pos = [0,0.2]U[0.4,0.6]U[0.8,1] 
        neg = [0.2,0.4]U[0.6,0.8]
        we will calculate E[(0-1)loss(h(X),Y)] by the total probabilty law
        """
        e = 0
        pos_sum = 0
        pos_segments = [[0,0.2],[0.4,0.6],[0.8,1]]
        for seg in pos_segments :
            pos_sum += self.calc_intersection(intervals, seg[0], seg[1])
        e += 0.2*pos_sum # false positive
        e += (0.6-pos_sum) * 0.8 # false prediction of 0 in pos_seg
        neg_sum = 0
        neg_segments = [[0.2,0.4],[0.6,0.8]]
        for seg in neg_segments :
            neg_sum += self.calc_intersection(intervals, seg[0], seg[1])
        e += 0.9*neg_sum # false positive
        e+= (0.4-neg_sum) * 0.1 # false prediction of 1 in neg_seg
        return e
    
    
    def calc_intersection(self, intervals, a, b):
        """
        The method returns the intersection of the intervals 
        with the segment [a, b]
        """
        sum = 0
        for interval in intervals:
            start = max(a,interval[0])
            end = min(b,interval[1])
            if(start<end):
                sum += (end-start)
        return sum 

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    # ass.draw_sample_intervals(100, 3) 
    # ass.experiment_m_range_erm(10, 40, 5, 3, 10)
    # ass.experiment_k_range_erm(100, 1, 10, 1)
    # ass.experiment_k_range_srm(1000, 1, 10, 1)
    ass.cross_validation(500, 3)


