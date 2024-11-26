import numpy as np

class Kalman:

    '''
    Implementaion of the EKF as described in the paper. All the theoretical details can be found in the report
    Vk: covariance matrix of the state
    Wk: covariance matrix of the input
    Pk: corrected covariance
    X_k_h: state of the robot
    '''

    def __init__(self,state0,t_start, print_info = False):
        self.print_info = print_info
        self.X_k_h = np.array(state0).T
        self.t_start = t_start
        self.Pk = np.zeros(4)


        var_v = 0.01
        var_w = 0.01
        self.Vk = np.eye(4) * var_v
        #self.Vk[2][2] = 0.3
        #self.Vk[3][3] = 0.3

        self.Wk = np.eye(2) * var_w
        
        if self.print_info:
            print("cov state")
            print(self.Vk)
            print("cov input")
            print(self.Wk)

    def predict_new_state(self,xk,dt):
        Fk = np.array([[1, 0, dt, 0], [0, 1, 0, dt],[0, 0, 1, 0],[0, 0, 0, 1]])
        xkp1 = np.matmul(Fk,xk) 
        return xkp1

    def predict(self, time, solo = False):
        delta_t = time - self.t_start
        self.t_start = time
        Fk = np.array([[1, 0, delta_t, 0], [0, 1, 0, delta_t],[0, 0, 1, 0],[0, 0, 0, 1]])
        self.X_kp1_pred_h = np.matmul(Fk,self.X_k_h)
        self.P_kp1_pred = np.matmul(Fk, np.matmul(self.Pk,Fk.T)) + self.Vk
        if self.print_info:
            print("predicted state:")
            print(self.X_kp1_pred_h) 
            print("predicted cov")
            print(self.P_kp1_pred)
            print("delta t")
            print(delta_t)

        if solo:
            self.X_k_h = self.X_kp1_pred_h
            self.Pk = self.P_kp1_pred

        return self.X_kp1_pred_h

    def correct(self,z_dx_kp1,z_dy_kp1):
        innovation_kp1 =  np.array([z_dx_kp1 - self.X_kp1_pred_h[0], z_dy_kp1 - self.X_kp1_pred_h[1]])

        H_kp1 = np.array([[1, 0 , 0, 0], [ 0, 1,0 ,0]])
        HPH = np.matmul(H_kp1,np.matmul(self.P_kp1_pred,H_kp1.T))
        if self.print_info:
            print("HPH", HPH)
        HPHW_inv = np.linalg.inv(HPH+self.Wk)

        if self.print_info:
            print("HPHW_INV",HPHW_inv)
            print("PHt", np.matmul(self.P_kp1_pred , H_kp1.T))
        R_kp1 = np.matmul(np.matmul(self.P_kp1_pred , H_kp1.T),HPHW_inv)

        if self.print_info:
            print("measured state:", z_dx_kp1,z_dy_kp1)
            print("innovation:", innovation_kp1)
            print("R times innovation:", np.matmul(R_kp1,innovation_kp1))
            print("kalman gain", R_kp1)

        X_kp1_h = self.X_kp1_pred_h + np.matmul(R_kp1,innovation_kp1)
        P_kp1 = self.P_kp1_pred - np.matmul(R_kp1, np.matmul(H_kp1, self.P_kp1_pred))
        self.X_k_h = X_kp1_h
        self.Pk = P_kp1

        if self.print_info:
            print("corrected state:", self.X_k_h)
            print("corrected cov:", self.Pk)
        return self.X_k_h, innovation_kp1
