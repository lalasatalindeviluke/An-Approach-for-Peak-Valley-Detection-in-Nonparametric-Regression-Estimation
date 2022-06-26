import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from scipy.optimize import minimize
from collections import deque
import statsmodels.api as sm
from math import *
from sklearn.isotonic import IsotonicRegression, check_increasing
from patsy import bs, dmatrix
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector, FloatVector
from quadprog import solve_qp
np.random.seed(42)
# 引入R套件
utils = rpackages.importr('utils')
base = importr('base')
# print(base.R_home())
splines = importr("splines")
R_quadprog = importr("quadprog")

# 建立自動查找 Peak Valley 程式
class Peak_Valley_Simu():
    def __init__(self):
        self.x = None
        self.y = None
        self.n = None
        self.left_dot_num = None
        self.right_dot_num = None
        self.left_start_index, self.left_end_index = None, None
        self.right_start_index, self.right_end_index = None, None
        self.intercept = None
    
    def generate_data(self, n, region, intercept, left_slope, right_slope, error_term):
        self.n = n
        self.x = np.linspace(-region, region, 2*n)
        y_left = intercept + left_slope*self.x[:n] + error_term*np.random.randn(n)
        y_right = intercept + right_slope*self.x[n:] + error_term*np.random.randn(n)
        self.y = np.hstack((y_left, y_right))
        self.beta_left = left_slope
        self.beta_right = right_slope
    
    def auto_peakzone_detection(self, x_coordinate, initial_range=0.01, iter_scale=1, maxiter=100, estimation_mode=False, quote=False):
        count = 0
        fit_range = initial_range

        while count < maxiter:
            self.left_start_index = np.min(np.where(self.x > x_coordinate-fit_range)[0])
            self.left_end_index = np.max(np.where(self.x < x_coordinate)[0])
            self.right_start_index = np.min(np.where(self.x > x_coordinate)[0])
            self.right_end_index = np.max(np.where(self.x < x_coordinate+fit_range)[0])
            
            X = sm.add_constant(self.x)

            self.left_dot_num = self.left_end_index-self.left_start_index+1
            self.right_dot_num = self.right_end_index-self.right_start_index+1

            mod_left = sm.OLS(self.y[self.left_start_index:self.left_end_index+1],
                            X[self.left_start_index:self.left_end_index+1])
            res_left = mod_left.fit()
            mod_right = sm.OLS(self.y[self.right_start_index:self.right_end_index+1],
                            X[self.right_start_index:self.right_end_index+1])
            res_right = mod_right.fit()

            left_Sxx = res_left.ess / (res_left.params[1])**2
            right_Sxx = res_right.ess / (res_right.params[1])**2

            if not estimation_mode:
                z_left = -t.ppf(0.05, df=self.left_dot_num-2) - self.beta_left / sqrt(res_left.mse_resid/left_Sxx)
                z_right = t.ppf(0.05, df=self.right_dot_num-2) - self.beta_right / sqrt(res_right.mse_resid/right_Sxx)
            if estimation_mode:
                z_left = -t.ppf(0.05, df=self.left_dot_num-2) - res_left.params[1] / sqrt(res_left.mse_resid/left_Sxx)
                z_right = t.ppf(0.05, df=self.right_dot_num-2) - res_right.params[1] / sqrt(res_right.mse_resid/right_Sxx)
            
            if ((1-norm.cdf(z_left) >= sqrt(0.9)) and (norm.cdf(z_right) >= sqrt(0.9)) and (res_left.pvalues[1]+res_right.pvalues[1])/2 <= 0.05):
                break
            
            # if (res_left.pvalues[1]+res_right.pvalues[1])/2 <= 0.05:
            #     break

            fit_range += iter_scale*initial_range
            count += 1

        if (count == maxiter and quote):
            return False
            # return {"fitting range":round(fit_range, 5), "iteration number":count,
            #         "left regression":res_left, "right regression":res_right,
            #         "z_left":z_left, "z_right":z_right,
            #         "left_Sxx":left_Sxx, "right_Sxx":right_Sxx}
        
        return {"fitting range":round(fit_range, 5), "iteration number":count,
                "left regression":res_left, "right regression":res_right,
                "z_left":z_left, "z_right":z_right,
                "left_Sxx":left_Sxx, "right_Sxx":right_Sxx}
                # "left index":[i for i in range(left_start_index, left_end_index+1)], "right index":[j for j in range(right_start_index, right_end_index+1)]
    
    def auto_valleyzone_detection(self, x_coordinate, initial_range=0.01, iter_scale=1, maxiter=100, estimation_mode=False, quote=False):
        count = 0
        fit_range = initial_range
        while count < maxiter:

            left_start_index = np.min(np.where(self.x > x_coordinate-fit_range)[0])
            left_end_index = np.max(np.where(self.x < x_coordinate)[0])
            right_start_index = np.min(np.where(self.x > x_coordinate)[0])
            right_end_index = np.max(np.where(self.x < x_coordinate+fit_range)[0])
            X = sm.add_constant(self.x)
            self.left_dot_num = left_end_index-left_start_index+1
            self.right_dot_num = right_end_index-right_start_index+1

            mod_left = sm.OLS(self.y[left_start_index:left_end_index+1],
                              X[left_start_index:left_end_index+1])
            res_left = mod_left.fit()
            mod_right = sm.OLS(self.y[right_start_index:right_end_index+1],
                               X[right_start_index:right_end_index+1])
            res_right = mod_right.fit()

            left_Sxx = res_left.ess / (res_left.params[1])**2
            right_Sxx = res_right.ess / (res_right.params[1])**2

            if not estimation_mode:
                z_left = t.ppf(0.05, df=self.left_dot_num-2) - self.beta_left / sqrt(res_left.mse_resid/left_Sxx)
                z_right = -t.ppf(0.05, df=self.right_dot_num-2) - self.beta_right / sqrt(res_right.mse_resid/right_Sxx)
            if estimation_mode:
                z_left = t.ppf(0.05, df=self.left_dot_num-2) - res_left.params[1] / sqrt(res_left.mse_resid/left_Sxx)
                z_right = -t.ppf(0.05, df=self.right_dot_num-2) - res_right.params[1] / sqrt(res_right.mse_resid/right_Sxx)

            if ((norm.cdf(z_left) >= sqrt(0.9)) and (1-norm.cdf(z_right) >= sqrt(0.9)) and (res_left.pvalues[1]+res_right.pvalues[1])/2 <= 0.05):
                break

            fit_range += iter_scale*initial_range
            count += 1
        
        if (count == maxiter and quote):
            return False

        return {"fitting range":round(fit_range, 5), "iteration number":count,
                "left regression":res_left, "right regression":res_right,
                "z_left":z_left, "z_right":z_right,
                "left_Sxx":left_Sxx, "right_Sxx":right_Sxx}

    def scatter_peakreg_plot(self, x_coordinate, estimation_mode):
        plt.figure(figsize=(16, 9))
        plt.grid()
        plt.scatter(self.x, self.y)
        plt.plot(self.x[self.left_start_index:self.left_end_index+1], # self.n-self.left_dot_num+1:self.n+1
                 self.auto_peakzone_detection(x_coordinate, estimation_mode=estimation_mode)["left regression"].predict(),
                 "r")
        plt.plot(self.x[self.right_start_index:self.right_end_index+1], # self.n+1:self.n+self.right_dot_num+1
                 self.auto_peakzone_detection(x_coordinate, estimation_mode=estimation_mode)["right regression"].predict(),
                 "k")
                 
    def scatter_valleyreg_plot(self, x_coordinate):
        plt.figure(figsize=(16, 9))
        plt.grid()
        plt.scatter(self.x, self.y)
        plt.plot(self.x[self.left_start_index:self.left_end_index+1],
                 self.auto_valleyzone_detection(x_coordinate)["left regression"].predict(),
                 "r")
        plt.plot(self.x[self.right_start_index:self.right_end_index+1],
                 self.auto_valleyzone_detection(x_coordinate)["right regression"].predict(),
                 "k")

    def simu_probability(self, left_slope, right_slope, condition="peak"):
        fitting_range_list = []
        left_dot_num_list, right_dot_num_list = [], []
        left_betahat_list, right_betahat_list = [], []
        leftsigma_hat_list, rightsigma_hat_list = [], []
        left_Sxx, right_Sxx = [], []

        for i in range(1000):
            self.generate_data(500, 1, 0, left_slope=left_slope, right_slope=right_slope, error_term=0.3)
            apd = self.auto_peakzone_detection(0)
            
            fitting_range_list.append(apd["fitting range"])
            left_dot_num_list.append(self.left_dot_num)
            right_dot_num_list.append(self.right_dot_num)
            left_betahat_list.append(apd["left regression"].params[1])
            right_betahat_list.append(apd["right regression"].params[1])
            leftsigma_hat_list.append(sqrt(apd["left regression"].mse_resid))
            rightsigma_hat_list.append(sqrt(apd["right regression"].mse_resid))
            left_Sxx.append(apd["left_Sxx"])
            right_Sxx.append(apd["right_Sxx"])

        fitting_range_list = np.array(fitting_range_list)
        left_dot_num_list = np.array(left_dot_num_list)
        right_dot_num_list = np.array(right_dot_num_list)
        left_betahat_list = np.array(left_betahat_list)
        right_betahat_list = np.array(right_betahat_list)
        leftsigma_hat_list = np.array(leftsigma_hat_list)
        rightsigma_hat_list = np.array(rightsigma_hat_list)
        left_Sxx = np.array(left_Sxx)
        right_Sxx = np.array(right_Sxx)

        # z_left = -t.ppf(0.05, df=self.left_dot_num-2) - beta_left / (np.mean(leftsigma_hat_list)/sqrt(left_Sxx))
        # z_right = t.ppf(0.05, df=self.right_dot_num-2) - beta_right / (np.mean(rightsigma_hat_list)/sqrt(right_Sxx))

        if condition == "peak":
            nonre_left_Z = left_betahat_list - (-t.ppf(0.05, df=left_dot_num_list-2))*(leftsigma_hat_list/np.sqrt(left_Sxx))
            nonre_right_Z = right_betahat_list + (-t.ppf(0.05, df=right_dot_num_list-2))*(rightsigma_hat_list/np.sqrt(right_Sxx))
            
            return {"non regularized left Z":nonre_left_Z, "non regularized right Z":nonre_right_Z,
                    "left index":np.where(nonre_left_Z > 0), "right index":np.where(nonre_right_Z < 0),
                    "beta1_hat array":left_betahat_list, "beta2_hat array":right_betahat_list,
                    "left sigma_hat array":leftsigma_hat_list, "right sigma_hat array":rightsigma_hat_list,
                    "fitting range array":fitting_range_list}
        else:
            nonre_left_Z = left_betahat_list + (-t.ppf(0.05, df=left_dot_num_list-2))*(leftsigma_hat_list/np.sqrt(left_Sxx))
            nonre_right_Z = right_betahat_list - (-t.ppf(0.05, df=right_dot_num_list-2))*(rightsigma_hat_list/np.sqrt(right_Sxx))

            return {"non regularized left Z":nonre_left_Z, "non regularized right Z":nonre_right_Z,
                    "left index":np.where(nonre_left_Z < 0), "right index":np.where(nonre_right_Z > 0),
                    "beta1_hat array":left_betahat_list, "beta2_hat array":right_betahat_list,
                    "left sigma_hat array":leftsigma_hat_list, "right sigma_hat array":rightsigma_hat_list,
                    "fitting range array":fitting_range_list}
    
    def auto_peak_points_detection_v3(self, step, distance, initial_range=0.01, iter_scale=1):
        '''
        假定 x, y 資料已經經過排序
        step: 一開始分割資料的步伐，用來判斷peak約略落在的位置
        distance: 選完初始分界點後，往左往右遍歷的candidate_points_list其每個點用來fit的最大範圍之一
        '''
        Full_range = max(self.x) - min(self.x)
        total_step = Full_range // step

        initial_candidate_points = [self.x[0] + i*step for i in range(int(total_step)+1)]
        real_candidate_points = [min(self.x, key=lambda k: abs(k-j)) for j in initial_candidate_points]
        
        candidate_points_dict = {}
        candidate_points_list = [] #待測試點座標
        for candidate_point in real_candidate_points: # 第一步先判斷peak粗略發生的位置
            try:
                candidate_dicts = self.auto_peakzone_detection(x_coordinate=candidate_point,
                                                               initial_range=initial_range, iter_scale=iter_scale,
                                                               estimation_mode=True, quote=True)
                if candidate_dicts:
                    candidate_points_dict[f"{candidate_point}"] = candidate_dicts
                    candidate_points_list.append(candidate_point)
            except:
                continue
        
        significant_point_deque = deque([],maxlen=1) # 收錄初始分界點座標
        significant_dict_deque = deque([],maxlen=1)
        for candidate_points_key, candidate_points_value in candidate_points_dict.items():
        # 找經過第一步中peak候選點裡最顯著(最短fitting range)的當初始點
            if len(significant_point_deque) == 0:
                significant_point_deque.append(float(candidate_points_key))
                significant_dict_deque.append(candidate_points_value)
            if significant_dict_deque[0]["iteration number"] > candidate_points_value["iteration number"]:
                significant_point_deque.append(float(candidate_points_key)) # 紀錄x座標(分界點)
                significant_dict_deque.append(candidate_points_value) # 紀錄x座標下的其他字典詳細資訊
        
        candidate_points_array = np.array(candidate_points_list)        
        
        approved_testing_deque = deque([significant_point_deque[0]])
        approved_testing_detail_deque = deque([significant_dict_deque[0]])
        # 往左檢查遍歷，找尋下一個分界點
        for untest_left in candidate_points_list[:candidate_points_list.index(significant_point_deque[0])][::-1]:
            maxiter = min(floor(((approved_testing_deque[0]-untest_left) - initial_range)*1000)/1000.0 / (initial_range*iter_scale),
                          floor((distance-initial_range)*1000)/1000.0 / (initial_range*iter_scale))
            
            approved_leftresult = self.auto_peakzone_detection(x_coordinate=untest_left, maxiter=int(maxiter),
                                                               initial_range=initial_range, iter_scale=iter_scale,
                                                               estimation_mode=True, quote=True)
            if approved_leftresult:
                approved_testing_deque.appendleft(untest_left)
                approved_testing_detail_deque.appendleft(approved_leftresult)
        # 往右檢查遍歷，找尋下一個分界點
        for untest_right in candidate_points_list[candidate_points_list.index(significant_point_deque[0])+1:]:
            maxiter = min(floor(((untest_right-approved_testing_deque[-1]) - initial_range)*1000)/1000.0 / (initial_range*iter_scale),
                          floor((distance-initial_range)*1000)/1000.0 / (initial_range*iter_scale))
            
            approved_rightresult = self.auto_peakzone_detection(x_coordinate=untest_right, maxiter=int(maxiter),
                                                                initial_range=initial_range, iter_scale=iter_scale,
                                                                estimation_mode=True, quote=True)
            if approved_rightresult:
                approved_testing_deque.append(untest_right)
                approved_testing_detail_deque.append(approved_rightresult)
        
        self.peak_coordinate = approved_testing_deque
        self.significant_point_deque = significant_point_deque
        return significant_point_deque, approved_testing_deque, approved_testing_detail_deque # 初始分界點，各peak座標，各座標fitting詳細
    
    def auto_valley_points_detection(self, step, distance, initial_range=0.01, iter_scale=1): # v3
        Full_range = max(self.x) - min(self.x)
        total_step = Full_range // step

        initial_candidate_points = [self.x[0] + i*step for i in range(int(total_step)+1)]
        real_candidate_points = [min(self.x, key=lambda k: abs(k-j)) for j in initial_candidate_points]
        
        candidate_points_dict = {}
        candidate_points_list = [] #待測試點座標
        for candidate_point in real_candidate_points: # 第一步先判斷peak粗略發生的位置
            try:
                candidate_dicts = self.auto_valleyzone_detection(x_coordinate=candidate_point,
                                                                 initial_range=initial_range, iter_scale=iter_scale,
                                                                 estimation_mode=True)
                if candidate_dicts:
                    candidate_points_dict[f"{candidate_point}"] = candidate_dicts
                    candidate_points_list.append(candidate_point)
            except:
                continue
        
        significant_point_deque = deque([],maxlen=1) # 收錄初始分界點座標
        significant_dict_deque = deque([],maxlen=1)
        for candidate_points_key, candidate_points_value in candidate_points_dict.items():
            if len(significant_point_deque) == 0:
                significant_point_deque.append(float(candidate_points_key))
                significant_dict_deque.append(candidate_points_value)
            if significant_dict_deque[0]["iteration number"] > candidate_points_value["iteration number"]:
                significant_point_deque.append(float(candidate_points_key)) # 紀錄x座標(分界點)
                significant_dict_deque.append(candidate_points_value) # 紀錄x座標下的其他字典詳細資訊
        
        candidate_points_array = np.array(candidate_points_list)
        
        
        approved_testing_deque = deque([significant_point_deque[0]])
        approved_testing_detail_deque = deque([significant_dict_deque[0]])
        # 往左檢查遍歷，找尋下一個分界點
        for untest_left in candidate_points_list[:candidate_points_list.index(significant_point_deque[0])][::-1]:
            maxiter = min(floor(((approved_testing_deque[0]-untest_left) - initial_range)*1000)/1000.0 / (initial_range*iter_scale),
                          floor((distance-initial_range)*1000)/1000.0 / (initial_range*iter_scale))
            approved_leftresult = self.auto_valleyzone_detection(x_coordinate=untest_left, maxiter=int(maxiter),
                                                                initial_range=initial_range, iter_scale=iter_scale,
                                                                estimation_mode=True, quote=True)
            if approved_leftresult:
                approved_testing_deque.appendleft(untest_left)
                approved_testing_detail_deque.appendleft(approved_leftresult)
        # 往右檢查遍歷，找尋下一個分界點
        for untest_right in candidate_points_list[candidate_points_list.index(significant_point_deque[0])+1:]:
            maxiter = min(floor(((untest_right-approved_testing_deque[-1]) - initial_range)*1000)/1000.0 / (initial_range*iter_scale),
                          floor((distance-initial_range)*1000)/1000.0 / (initial_range*iter_scale))
            approved_rightresult = self.auto_valleyzone_detection(x_coordinate=untest_right, maxiter=int(maxiter),
                                                                  initial_range=initial_range, iter_scale=iter_scale,
                                                                  estimation_mode=True, quote=True)
            if approved_rightresult:
                approved_testing_deque.append(untest_right)
                approved_testing_detail_deque.append(approved_rightresult)
        
        self.valley_coordinate = approved_testing_deque
        return significant_point_deque, approved_testing_deque, approved_testing_detail_deque # 初始分界點，各valley座標，各座標fitting詳細
    
    def peak_valley_index(self):
        pv_coordinate = sorted(list(self.peak_coordinate + self.valley_coordinate))
        pv_index = [np.where(self.x==i)[0][0] for i in pv_coordinate]
        pv_index.insert(0, 0)
        pv_index.append(len(self.x)-1)
        self.pv_coordinate = pv_coordinate
        self.pv_index = pv_index
        return pv_coordinate, pv_index
    
    def isotonic_reg_rss(self, pv_coordinate_diff_log_odds_form):
        '''
            pv_coordinate_diff_log_odds_form: np.array
            會需要用到 self.pv_coordinate ，所以要確保先呼叫 peak_valley_index() 方法
        '''
        pv_coordinate_copy = np.hstack((self.x[0], np.array(self.pv_coordinate), self.x[-1]))
        pv_coordinate_copy2 = pv_coordinate_copy[1:] - pv_coordinate_copy[:-1]
        mean = pv_coordinate_copy2.mean()
        pv_coordinate_diff = np.exp(pv_coordinate_diff_log_odds_form) * mean
        pv_refactor_coordinate = pv_coordinate_diff.cumsum()[:-1]
        nearest_x = [min(self.x, key=lambda k: abs(k-j)) for j in pv_refactor_coordinate]
        nearest_x_index = [list(self.x).index(i) for i in nearest_x]
        nearest_x_index = [0] + nearest_x_index + [999]

        y_dict = {}
        for i in range(len(nearest_x_index)-1):
            if i == len(nearest_x_index)-2:
                increasing = check_increasing(self.x[nearest_x_index[i]:], self.y[nearest_x_index[i]:])
                ir = IsotonicRegression(increasing=increasing)
                y_dict[f"y_{nearest_x_index[i]}_{nearest_x_index[i+1]}"] = ir.fit_transform(self.x[nearest_x_index[i]:], self.y[nearest_x_index[i]:])
            else:
                increasing = check_increasing(self.x[nearest_x_index[i]:nearest_x_index[i+1]], self.y[nearest_x_index[i]:nearest_x_index[i+1]])
                ir = IsotonicRegression(increasing=increasing)
                y_dict[f"y_{nearest_x_index[i]}_{nearest_x_index[i+1]}"] = ir.fit_transform(self.x[nearest_x_index[i]:nearest_x_index[i+1]],
                                                                                            self.y[nearest_x_index[i]:nearest_x_index[i+1]])
        
        y_predicted = np.hstack([y_dict[list(y_dict.keys())[i]] for i in range(len(y_dict))])
        rss = ((self.y - y_predicted)**2).sum()
        return rss
    
    def isotonic_reg_rss_v2(self, pv_coordinate_softmax_form): # 使用softmax
        '''pv_coordinate_softmax_form: np.array'''
        pv_coordinate_original_form \
        = self.significant_point_deque[0] \
            * (1/pv_coordinate_softmax_form[self.pv_coordinate.index(self.significant_point_deque[0])]) \
            * pv_coordinate_softmax_form
        
        nearest_x = [min(self.x, key=lambda k: abs(k-j)) for j in pv_coordinate_original_form]
        nearest_x_index = [list(self.x).index(i) for i in nearest_x]
        nearest_x_index = [0] + nearest_x_index + [999]

        y_dict = {}
        for i in range(len(nearest_x_index)-1):
            if i == len(nearest_x_index)-2:
                increasing = check_increasing(self.x[nearest_x_index[i]:], self.y[nearest_x_index[i]:])
                ir = IsotonicRegression(increasing=increasing)
                y_dict[f"y_{nearest_x_index[i]}_{nearest_x_index[i+1]}"] = ir.fit_transform(self.x[nearest_x_index[i]:], self.y[nearest_x_index[i]:])
            else:
                increasing = check_increasing(self.x[nearest_x_index[i]:nearest_x_index[i+1]], self.y[nearest_x_index[i]:nearest_x_index[i+1]])
                ir = IsotonicRegression(increasing=increasing)
                y_dict[f"y_{nearest_x_index[i]}_{nearest_x_index[i+1]}"] = ir.fit_transform(self.x[nearest_x_index[i]:nearest_x_index[i+1]],
                                                                                            self.y[nearest_x_index[i]:nearest_x_index[i+1]])
        
        y_predicted = np.hstack([y_dict[list(y_dict.keys())[i]] for i in range(len(y_dict))])
        rss = ((self.y - y_predicted)**2).sum()
        return rss
    
    def isotonic_reg_rss_v3(self, pv_coordinate_softmax_form): # 使用softmax
        '''pv_coordinate_softmax_form: np.array'''
        pv_coordinate_base_form = 1 / np.sum(np.exp(pv_coordinate_softmax_form))
        pv_coordinate_original_form = np.hstack((pv_coordinate_base_form, pv_coordinate_base_form*np.exp(pv_coordinate_softmax_form[1:]))).cumsum() * (self.x[-1]-self.x[0])
        pv_coordinate_original_form = pv_coordinate_original_form[:-1]
        
        nearest_x = [min(self.x, key=lambda k: abs(k-j)) for j in pv_coordinate_original_form]
        nearest_x_index = [list(self.x).index(i) for i in nearest_x]
        nearest_x_index = [0] + nearest_x_index + [999]
        
        y_dict = {}
        for i in range(len(nearest_x_index)-1):
            if i == len(nearest_x_index)-2:
                increasing = check_increasing(self.x[nearest_x_index[i]:], self.y[nearest_x_index[i]:])
                ir = IsotonicRegression(increasing=increasing)
                y_dict[f"y_{nearest_x_index[i]}_{nearest_x_index[i+1]}"] = ir.fit_transform(self.x[nearest_x_index[i]:], self.y[nearest_x_index[i]:])
            else:
                increasing = check_increasing(self.x[nearest_x_index[i]:nearest_x_index[i+1]], self.y[nearest_x_index[i]:nearest_x_index[i+1]])
                ir = IsotonicRegression(increasing=increasing)
                y_dict[f"y_{nearest_x_index[i]}_{nearest_x_index[i+1]}"] = ir.fit_transform(self.x[nearest_x_index[i]:nearest_x_index[i+1]],
                                                                                            self.y[nearest_x_index[i]:nearest_x_index[i+1]])
        
        y_predicted = np.hstack([y_dict[list(y_dict.keys())[i]] for i in range(len(y_dict))])
        rss = ((self.y - y_predicted)**2).sum()
        return rss

class IsotonicReg():
    def __init__(self, x, y, pv_index):
        self.x, self.y = x, y
        self.pv_index = pv_index
        self.rss = None
        self.y_predicted = None
    
    def iso_fit_transform(self):
        y_dict = {}
        for i in range(len(self.pv_index)-1):
            if i == len(self.pv_index)-2:
                increasing = check_increasing(self.x[self.pv_index[i]:], self.y[self.pv_index[i]:])
                ir = IsotonicRegression(increasing=increasing)
                y_dict[f"y_{self.pv_index[i]}_{self.pv_index[i+1]}"] = ir.fit_transform(self.x[self.pv_index[i]:], self.y[self.pv_index[i]:])
            else:
                increasing = check_increasing(self.x[self.pv_index[i]:self.pv_index[i+1]], self.y[self.pv_index[i]:self.pv_index[i+1]])
                ir = IsotonicRegression(increasing=increasing)
                y_dict[f"y_{self.pv_index[i]}_{self.pv_index[i+1]}"] = ir.fit_transform(self.x[self.pv_index[i]:self.pv_index[i+1]],
                                                                                        self.y[self.pv_index[i]:self.pv_index[i+1]])
        self.y_predicted = np.hstack([y_dict[list(y_dict.keys())[i]] for i in range(len(y_dict))])

    def isotonic_rss(self):
        self.rss = ((self.y - self.y_predicted)**2).sum()
        return self.rss
    
    def isotic_AIC(self, n, k):
        return n*log(self.rss/n) + 2*k
    
    def isotic_BIC(self, n, k):
        return n*log(self.rss/n) + k*log(n)
    
    def plot_isoreg(self):
        plt.figure(figsize=(16, 9), dpi=200)
        plt.scatter(self.x, self.y, s=10)
        plt.scatter(self.x[self.pv_index[1:-1]], self.y[self.pv_index[1:-1]], s=100, c="r", marker='*')
        plt.plot(self.x, self.y_predicted, c="tab:orange", linewidth=3)
        plt.grid()
        plt.show()


# 下面這個class裡解二次限制下的splines迴歸是錯的，主要從此class得到backward elimination出來的節點。
# 錯誤地方在沒有把meq該等於0的位置搬到矩陣前方。
class constrained_splines_reg():
    def __init__(self, x, y, pv_index):
        self.x, self.y = x, y
        self.pv_index = pv_index
        self.rss = None
        self.y_predicted = None
        self.new_knots = None
    
    def __equidistant_knots(self, n=10):
        knots_for_each_part = []
        for i in range(len(self.pv_index)-1):
            knots_for_each_part.append(list(np.linspace(self.x[self.pv_index[i]], self.x[self.pv_index[i+1]], n+1)[1:-1]))
        return knots_for_each_part # 等距切出節點
    
    def findall_splines_reg_with_backward_elimination(self):
        knots_for_each_part = self.__equidistant_knots()
        bx = {}
        BX = {}
        new_knots = []
        bspline_basis_knots = []
        for i in range(len(self.pv_index)-1):
            if i == len(self.pv_index)-2:
                bx[f"bx{i}"] = bs(x=self.x[self.pv_index[i]:], knots=knots_for_each_part[i],
                                  lower_bound=self.x[self.pv_index[i]], upper_bound=self.x[self.pv_index[i+1]],
                                  include_intercept=True, degree=2)
            else:
                bx[f"bx{i}"] = bs(x=self.x[self.pv_index[i]:self.pv_index[i+1]], knots=knots_for_each_part[i],
                                  lower_bound=self.x[self.pv_index[i]], upper_bound=self.x[self.pv_index[i+1]],
                                  include_intercept=True, degree=2)
            temp_bx = bx[f"bx{i}"].copy()
            temp_knots = knots_for_each_part[i].copy()
            condition = True
            while condition:
                if i == len(self.pv_index)-2:
                    mod = sm.OLS(self.y[self.pv_index[i]:], temp_bx)
                    result = mod.fit()
                else:
                    mod = sm.OLS(self.y[self.pv_index[i]:self.pv_index[i+1]], temp_bx)
                    result = mod.fit()
                if result.pvalues.max() > 0.05:
                    temp_bx = np.delete(temp_bx, result.pvalues.argmax(), axis=1)
                    temp_knots.pop(result.pvalues.argmax())
                else:
                    condition = False
            BX[f"BX{i}"] = temp_bx
            new_knots.append(temp_knots)
            bspline_basis_knots.append([self.x[self.pv_index[i]]]*3 + temp_knots + [self.x[self.pv_index[i+1]]]*3)
        self.new_knots = new_knots
        return bx, BX, new_knots, bspline_basis_knots # 初始bspline design matrix, 修正節點後的bspline design matrix, 修正後顯著節點, 節點+頭尾*ord次數

    def findall_splineDesignMatrix(self, deg=2): # 如果資料結構不要用dictionary，而是直接矩陣表示更好
        bx, BX, new_knots, bspline_basis_knots = self.findall_splines_reg_with_backward_elimination()
        splineDesignMatrix_dict = {}
        for i, knots_list in enumerate(bspline_basis_knots):
            for j in range(len(knots_list) - 3):
                if j == 0:
                    basis_func_matrix = splines.splineDesign(knots=FloatVector(knots_list[j:j+4]),
                                                             x=FloatVector(knots_list[deg:-deg]),
                                                             derivs=1, ord=deg+1, outer_ok=True)
                    basis_func_matrix = np.array(basis_func_matrix)
                else:
                    temp_matrix = np.array(splines.splineDesign(knots=FloatVector(knots_list[j:j+4]),
                                                                x=FloatVector(knots_list[deg:-deg]),
                                                                derivs=1, ord=deg+1, outer_ok=True))
                    basis_func_matrix = np.hstack((basis_func_matrix, temp_matrix))
            
            splineDesignMatrix_dict[f"basis func matrix {i}"] = basis_func_matrix
        return splineDesignMatrix_dict # 一次微分後的bspline design matrix
    
    def generate_GaCb_matrix(self, Bsplines_dmatrix, SplineDesign_Derivative_matrix, y_part):
        '''
        Bsplines_dmatrix: 修正節點後的bspline design matrix
        SplineDesign_Derivative_matrix: 一次微分後的bspline design matrix
        y_part: 根據peak valley索引位置切出的各段y
        '''
        G = 2 * Bsplines_dmatrix.T @ Bsplines_dmatrix
        a = (2* Bsplines_dmatrix.T @ y_part) # 0軸向量
        C = SplineDesign_Derivative_matrix.T
        # b = np.ones((C.shape[1], ))
        b = np.zeros((C.shape[1], ))
        return G, a, C, b
    
    def Solve_QP(self, meq=0, first_part_increase=True):
        sign = 1 if first_part_increase else -1

        knots_for_each_part = self.__equidistant_knots()
        _, BX_dict, _, bspline_basis_knots= self.findall_splines_reg_with_backward_elimination()
        splineDesignMatrix_dict = self.findall_splineDesignMatrix()
        constrained_solution = []
        for part in range(len(self.pv_index)-1):
            y_part = self.y[self.pv_index[len(self.pv_index)-2]:] if part==len(self.pv_index)-2 else self.y[self.pv_index[part]:self.pv_index[part+1]]
            G, a, C, b = self.generate_GaCb_matrix(list(BX_dict.values())[part], list(splineDesignMatrix_dict.values())[part], y_part)
            if part%2 == 0:
                solution_, value_, unconstrained_solution_, iterations_, Lagrangian_, iact_ = solve_qp(G, a, C*sign, b, meq=meq)
            else:
                solution_, value_, unconstrained_solution_, iterations_, Lagrangian_, iact_ = solve_qp(G, a, -C*sign, b, meq=meq)
            constrained_solution.append(solution_)

        return constrained_solution
    
    def fit_constrained_transform(self, meq=0, first_part_increase=True):
        _, BX_dict, _, bspline_basis_knots= self.findall_splines_reg_with_backward_elimination()
        constrained_solution = self.Solve_QP(meq=meq, first_part_increase=first_part_increase)

        for part in range(len(self.pv_index)-1):
            if part == 0:
                y_bs_constrained_result = (list(BX_dict.values())[part] @ constrained_solution[part].reshape((-1, 1))).reshape((-1, ))
            else:
                temp_y_bs_constrained_result = (list(BX_dict.values())[part] @ constrained_solution[part].reshape((-1, 1))).reshape((-1, ))
                y_bs_constrained_result = np.hstack((y_bs_constrained_result, temp_y_bs_constrained_result))
        return y_bs_constrained_result


class comprehensive_csr():
    def __init__(self, x, y, pv_coordinate, pv_index, knots_of_each_part):
        '''
        pv: coordinate of peak & valley
        '''
        self.x = x
        self.y = y
        self.pv = pv_coordinate if type(pv_coordinate)==list else list(pv_coordinate) # not include head & tail
        self.pv_index = pv_index # include head & tail
        self.knots_of_each_part = knots_of_each_part if type(knots_of_each_part)==list else list(knots_of_each_part) # 只有各段節點(不含peak、valley)
        
    def comprehensive_knots(self):
        comprehensive_knots = self.knots_of_each_part + self.pv
        comprehensive_knots.sort()

        knots_ht = [self.x[0]] + comprehensive_knots + [self.x[-1]]
        self.pv_index_in_compre_knots = []
        for i in self.pv:
            self.pv_index_in_compre_knots.append(knots_ht.index(i))

        return comprehensive_knots
    
    # def __comprehensive_parts_of_knots(self):
    #     y_part_dict = {}
    #     piecewise_knots = {}
    #     for i in range(len(self.pv_index) - 1):
    #         if i == len(self.pv_index) - 2:
    #             y_part_dict[f"part{i}"] = self.y[self.pv_index[i]:]
    #         y_part_dict[f"part{i}"] = self.y[self.pv_index[i]:self.pv_index[i+1]]

    #     comprehensive_knots = self.__comprehensive_knots()
    #     for i in range(len(self.pv_index_in_compre_knots) - 1):
    #         if i == len(self.pv_index_in_compre_knots) -2:
    #             piecewise_knots[f"part{0}"] = comprehensive_knots[self.pv_index_in_compre_knots[i]:]
    #         piecewise_knots[f"part{0}"] = comprehensive_knots[self.pv_index_in_compre_knots[i]:self.pv_index_in_compre_knots[i+1]]
    #     return piecewise_knots, y_part_dict

    def splineDesign_derivative_func_Matrix(self, deg):
        bspline_basis_knots = self.comprehensive_knots()
        bspline_basis_knots = [self.x[0]]*(deg+1) + bspline_basis_knots + [self.x[-1]]*(deg+1)

        for i in range(len(bspline_basis_knots) - (deg+1)):
            if i == 0:
                basis_derivative_func_matrix = splines.splineDesign(knots=FloatVector(bspline_basis_knots[i:i+deg+2]),
                                                                    x=FloatVector(bspline_basis_knots[deg:-deg]),
                                                                    derivs=1, ord=deg+1, outer_ok=True)
                basis_derivative_func_matrix = np.array(basis_derivative_func_matrix).reshape((-1, 1))
            else:
                temp_matrix = np.array(splines.splineDesign(knots=FloatVector(bspline_basis_knots[i:i+deg+2]),
                                                            x=FloatVector(bspline_basis_knots[deg:-deg]),
                                                            derivs=1, ord=deg+1, outer_ok=True)).reshape((-1, 1))
                basis_derivative_func_matrix = np.hstack((basis_derivative_func_matrix, temp_matrix))

        return basis_derivative_func_matrix # 一次微分後的bspline design matrix
    
    def splineDesign_derivative_xi_Matrix(self, xi, deg):
        bspline_basis_knots = self.comprehensive_knots()
        bspline_basis_knots = [self.x[0]]*(deg+1) + bspline_basis_knots + [self.x[-1]]*(deg+1)

        for i in range(len(bspline_basis_knots) - (deg+1)):
            if i == 0:
                basis_derivative_func_matrix = splines.splineDesign(knots=FloatVector(bspline_basis_knots[i:i+deg+2]),
                                                                    x=FloatVector(xi),
                                                                    derivs=1, ord=deg+1, outer_ok=True)
                basis_derivative_func_matrix = np.array(basis_derivative_func_matrix)
            else:
                temp_matrix = np.array(splines.splineDesign(knots=FloatVector(bspline_basis_knots[i:i+deg+2]),
                                                            x=FloatVector(xi),
                                                            derivs=1, ord=deg+1, outer_ok=True))
                basis_derivative_func_matrix = np.hstack((basis_derivative_func_matrix, temp_matrix))

        return basis_derivative_func_matrix # 一次微分後的bspline design matrix

    def generate_GaCb_matrix(self, deg, first_part_increase=True):
        '''
            Bsplines_dmatrix: bspline design matrix
            SplineDesign_Derivative_matrix: 一次微分後的bspline design matrix
            y_part: 根據peak valley索引位置切出的各段y
        '''
        SplineDesign_Derivative_matrix = self.splineDesign_derivative_func_Matrix(deg=deg)
        Bsplines_dmatrix = bs(x=self.x, knots=self.comprehensive_knots(), degree=deg,
                              include_intercept=True, lower_bound=self.x[0], upper_bound=self.x[-1])
        divide_points = [0] + self.pv_index_in_compre_knots + [SplineDesign_Derivative_matrix.shape[0] - 1] # 完整節點加上頭尾座標
        
        if first_part_increase:
            sign_list = [-1]*SplineDesign_Derivative_matrix.shape[0]
            tmp_counter = len(sign_list)

            if len(divide_points)%2 == 0: 
                for index in range(0, len(divide_points), 2):
                    if (index == 0):
                        sign_list[:divide_points[index+1]+1] = [1]*(divide_points[index+1]+1 - divide_points[index])
                    elif (index == len(divide_points)-2):
                        sign_list[divide_points[index]+1:] = [1]*(divide_points[-1] - (divide_points[index]+1) + 1)
                    else:
                        sign_list[divide_points[index]+1:divide_points[index+1]+1] = [1]*(divide_points[index+1]+1 - (divide_points[index]+1))
            else:
                for index in range(0, len(divide_points)-1, 2):
                    if (index == 0):
                        sign_list[:divide_points[index+1]+1] = [1]*(divide_points[index+1]+1 - divide_points[index])
                    else:
                        sign_list[divide_points[index]+1:divide_points[index+1]+1] = [1]*(divide_points[index+1]+1 - (divide_points[index]+1))

        else:
            sign_list = [1]*SplineDesign_Derivative_matrix.shape[0]
            tmp_counter = len(sign_list)

            if len(divide_points)%2 == 0:
                for index in range(0, len(divide_points), 2):
                    if (index == 0):
                        sign_list[:divide_points[index+1]] = [-1]*(divide_points[index+1] - divide_points[index])
                    elif (index == len(divide_points)-2):
                        sign_list[divide_points[index]+1:] = [-1]*(divide_points[-1] - (divide_points[index]+1) + 1)
                    else:
                        sign_list[divide_points[index]+1:divide_points[index+1]] = [-1]*(divide_points[index+1] - (divide_points[index]+1))
            else:
                for index in range(0, len(divide_points)-1, 2):
                    if (index == 0):
                        sign_list[:divide_points[index+1]] = [-1]*(divide_points[index+1] - divide_points[index])
                    else:
                        sign_list[divide_points[index]+1:divide_points[index+1]] = [-1]*(divide_points[index+1] - (divide_points[index]+1))
        
        
        G = 2 * Bsplines_dmatrix.T @ Bsplines_dmatrix
        a = (2 * self.y.reshape((-1,1)).T @ Bsplines_dmatrix).reshape((-1,))
        
        # a = (2 * Bsplines_dmatrix.T @ self.y.reshape((-1,1))).reshape((-1,)) # 0軸向量
        assert tmp_counter == len(sign_list)
        C = (np.diag(sign_list) @ SplineDesign_Derivative_matrix).T

        C_copy = C.copy()
        C_copy = np.delete(C_copy, self.pv_index_in_compre_knots, axis=1)
        C_meq = C[:, self.pv_index_in_compre_knots]
        C_final = np.hstack((C_meq, C_copy))

        b = np.zeros((C_final.shape[1], ))
        return G, a, C_final, b

    def Solve_QP(self, deg, meq=0, first_part_increase=True):
        # bspline_basis_knots = self.comprehensive_knots()
        G, a, C, b = self.generate_GaCb_matrix(first_part_increase=first_part_increase, deg=deg)

        constrained_solution, value_, unconstrained_solution_, iterations_, Lagrangian_, iact_ = solve_qp(G, a, C, b, meq=meq)
        return constrained_solution
    
    def fit_constrained_transform(self, deg, meq=0, first_part_increase=True):
        constrained_solution = self.Solve_QP(deg=deg, meq=meq, first_part_increase=first_part_increase)
        comprehensive_knots = self.comprehensive_knots()

        comprehensive_bx = bs(x=self.x, knots=comprehensive_knots, degree=deg, include_intercept=True, lower_bound=self.x[0], upper_bound=self.x[-1])
        return comprehensive_bx @ constrained_solution.reshape((-1,1))
    
    def fit_transform(self, deg=2):
        comprehensive_knots = self.comprehensive_knots()
        comprehensive_bx = bs(x=self.x, knots=comprehensive_knots, degree=deg, include_intercept=True, lower_bound=self.x[0], upper_bound=self.x[-1])

        model = sm.OLS(self.y, comprehensive_bx)
        result = model.fit()
        return comprehensive_bx @ result.params.reshape((-1,1))
 
