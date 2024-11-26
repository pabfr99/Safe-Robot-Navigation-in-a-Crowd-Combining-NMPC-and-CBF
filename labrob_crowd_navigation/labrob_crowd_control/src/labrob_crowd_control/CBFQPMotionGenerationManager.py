import math
import threading
import rospy
import geometry_msgs.msg
import tf2_ros
import labrob_crowd_navigation_utils.utils
import labrob_crowd_navigation_msgs.msg
import numpy as np
from quadprog import solve_qp

from labrob_crowd_control.MotionGenerationManager import *

class CBFQPMotionGenerationManager(MotionGenerationManager):
    def __init__(self):
        MotionGenerationManager.__init__(self)

        self.K_CLUSTERS = 3
        CBF_RHO = 0.8
        CBF_DS = 0.5
        self.CBF_RADIUS = (CBF_RHO + CBF_DS)

        self.CBF_ALPHA = 0.5
        self.TIAGO_A = 0.25
        self.TARGET_POINT = [5,5]

        self.MAXV = 1.0
        self.MAXV_NEG = -0.2

        DEG2RAD =  0.01745
        self.MAXW =  DEG2RAD * 120
        self.MAXW_NEG = -DEG2RAD  * 120

        self.REGULATION_K1 = 0.2
        self.REGULATION_K2 = 0.2
        self.REGULATION_K3 = 0.2

        self.TASK_TYPE = '8'

        self.prev_input = np.array([0.0, 0.0])
        self.timezero = 0

        self.target_switched = self.TARGET_POINT

    def saturate_input(self,input):
        if(input[0] > self.MAXV):
            input[0] = self.MAXV
        elif(input[0] < self.MAXV_NEG):
            input[0] = self.MAXV_NEG

        if(input[1] > self.MAXW):
            input[1] = self.MAXW
        elif(input[1] < self.MAXW_NEG):
            input[1] = self.MAXW_NEG
        return input

    def executeEightTraj(self,state,k1,k2,time):
        a = 5
        tiagox = state.x
        tiagoy = state.y
        tiagotheta = state.theta

        w = .1
        t= ((time-self.timezero) % (2*np.pi/w))

        x = a*np.cos(t*w) #a*np.sin(2*t*w) + xc
        dx = -a*w*np.sin(t*w)

        y = a*np.sin(w*t)
        dy = a*w*np.cos(t*w)

        #thetap=math.atan2(dy,dx)
        u1 = dx + k1*(x - (tiagox +(self.TIAGO_A*np.cos(tiagotheta))))
        u2 = dy + k2*(y - (tiagoy + (self.TIAGO_A*np.sin(tiagotheta))))
        vp = u1*np.cos(tiagotheta) + u2*np.sin(tiagotheta)
        wp = -u1*(np.sin(tiagotheta)/self.TIAGO_A) + u2*(np.cos(tiagotheta)/self.TIAGO_A)

        driving_velocity = vp  
        steering_velocity = wp

        return np.array((driving_velocity,steering_velocity))

    def regulation(self,state,target,k1,k2,k3,tol):
        '''
        Simple method to perform regularization provided during the lectures
        '''
        x = state.x
        y = state.y
        theta = state.theta
      
        # Desired configuration:
        try:
            x_d = target.position.x
            y_d = target.position.y
        except:
            x_d = target[0]
            y_d = target[1]
    
        theta_d = 0

        # Unicycle configuration in desired reference frame coordinates:
        x_r = math.cos(-theta_d) * (x - x_d) - math.sin(-theta_d) * (y - y_d)
        y_r = math.sin(-theta_d) * (x - x_d) + math.cos(-theta_d) * (y - y_d)
        theta_r = labrob_crowd_navigation_utils.utils.wrap_angle(-theta_d + theta)

        # Polar coordinates in relative coordinates:
        rho   = math.sqrt(math.pow(x_r, 2.0) + math.pow(y_r, 2.0))
        gamma = labrob_crowd_navigation_utils.utils.wrap_angle(math.atan2(y_r, x_r) - theta_r + math.pi)
        delta = labrob_crowd_navigation_utils.utils.wrap_angle(gamma + theta_r)

        # Feedback control:
        if abs(rho) < tol:
            driving_velocity = 0.0
            steering_velocity = 0.0
            swap = True

        else:
            driving_velocity  = k1 * rho * math.cos(gamma)
            steering_velocity = k2 * gamma + k1 * math.sin(gamma) * math.cos(gamma) / gamma * (gamma + k3 * delta)
            swap = False
        return np.array((driving_velocity,steering_velocity)),swap

    def updateCBF(self,input,old_vel,state,obs_list,cbf_radius,cbf_alpha,a,time):
        num_obstacles = len(obs_list)
        input_size = 2
        G = np.zeros((num_obstacles,input_size))
        b = np.zeros(num_obstacles)

        # Tiago state
        x = state.x
        y = state.y
        theta = state.theta

        h_save_vec = np.ones(self.K_CLUSTERS+2) * 999
        dh_save_vec = np.ones(self.K_CLUSTERS+2) * 999

        h_save_vec[-2] = time
        dh_save_vec[-2] = time

        for i in range(0,num_obstacles):
            xo = obs_list[i].position.x
            yo = obs_list[i].position.y
            dxo = obs_list[i].velocity.x
            dyo = obs_list[i].velocity.y            
           
            ha = (x - xo + a*math.cos(theta))**2 + (y - yo + a*math.sin(theta))**2 - cbf_radius**2
            g_v = (2*math.cos(theta)*(x - xo + a*math.cos(theta)) + 2*math.sin(theta)*(y - yo + a*math.sin(theta)))#*v 
            g_w = (2*a*math.cos(theta)*(y - yo + a*math.sin(theta)) - 2*a*math.sin(theta)*(x - xo + a*math.cos(theta)))#*w
            extra =  (- 2*dxo*(x - xo + a*math.cos(theta)) - 2*dyo*(y - yo + a*math.sin(theta)))
            extra = cbf_alpha*ha + extra
            extra = -1.0*extra

            b[i] = np.array(extra,dtype='d')
            G[i] = np.array((np.array(g_v),np.array(g_w)),dtype='d')

        P = np.array(([1,0],[0,1e-1]),dtype='d') #minimize velocity more
        q = np.matmul(P,input)
        negative_vbound_left = np.array([1, 0])
        negative_vbound_right = np.array([self.MAXV_NEG])

        negative_wbound_left = np.array([0, 1])
        negative_wbound_right = np.array([self.MAXW_NEG])

        positive_vbound_left = np.array([-1, 0])
        positive_vbound_right = np.array([-self.MAXV])

        positive_wbound_left = np.array([0, -1])
        positive_wbound_right = np.array([-self.MAXW])

        G = np.vstack((G,negative_vbound_left, negative_wbound_left, positive_vbound_left, positive_wbound_left))
        b = np.append(b,[negative_vbound_right, negative_wbound_right,positive_vbound_right, positive_wbound_right])

        try:
            sol, f, xu, iters, lagr, iact = solve_qp(P, q, G.T, b)
            h_save_vec[-1] = 0
            dh_save_vec[-1] = 0
        except ValueError:
            rospy.loginfo("WARNING! CONSTRAINT INCONSISTENT OR QP ERROR!")
            sol = np.array((0.0,0.0))
            h_save_vec[-1] = 1
            dh_save_vec[-1] = 1

        mod_input = sol

        return mod_input

    def update(self):
        obs_list = []
        # iterate thru the received obstacles from motion prediction module
        for motion_prediction in self.crowd_motion_prediction_stamped_rt.crowd_motion_prediction.motion_predictions:
            if(np.array_equal(np.array([motion_prediction.position.x,
                motion_prediction.position.y]), np.array([999,999]))):
                pass
            else:
                obs_list.append(motion_prediction)
        
        # decide a ff-input vector
        ff_input = np.array([0,0])
        if self.TASK_TYPE == 'REG':
            ff_input,swap = self.regulation(self.robot_configuration,self.target_switched,self.REGULATION_K1,self.REGULATION_K2,self.REGULATION_K3,0.5)      

        if self.TASK_TYPE == 'FOLLOW':
            if(len(obs_list) > 0):
                ff_input = self.regulation(self.robot_configuration,obs_list[0],1,1,1,0.01)

        if self.TASK_TYPE == '8':
            ff_input = self.executeEightTraj(self.robot_configuration,.12,.12,rospy.get_time())

        if self.TASK_TYPE == 'STILL':
            ff_input = np.array([0,0])

        old_ff_input = ff_input
        ff_input = self.saturate_input(ff_input)
        if not(np.array_equal(old_ff_input,ff_input)):
            rospy.loginfo("SATURATION!")
            exit()


        #update velocities
        driving_velocity = ff_input[0]
        steering_velocity = ff_input[1]

        #if there are obstacles, then apply cbf
        if(len(obs_list) > 0):

            # save the starting time for the reference trajectory
            if(self.timezero == 0):
                self.timezero = rospy.get_time()

            new_input = self.updateCBF(ff_input,self.prev_input,self.robot_configuration,obs_list,self.CBF_RADIUS,self.CBF_ALPHA,self.TIAGO_A,rospy.get_time())
            new_input = self.saturate_input(new_input)
            driving_velocity = new_input[0]
            steering_velocity = new_input[1]
            self.prev_input = np.array((driving_velocity,steering_velocity))

            # Fill unicycle control input:
            self.unicycle_control_input.driving_velocity = driving_velocity
            self.unicycle_control_input.steering_velocity = steering_velocity

def main():
    rospy.init_node('CBFQPMotionGenerationManager', log_level=rospy.INFO)
    
    motion_generation_manager = CBFQPMotionGenerationManager()
    motion_generation_manager.start()
