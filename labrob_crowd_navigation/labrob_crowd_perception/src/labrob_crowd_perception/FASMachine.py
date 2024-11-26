#from statemachine import StateMachine, State
import numpy as np

from labrob_crowd_perception.Kalman import * 
import labrob_crowd_perception.CommonVars as CommonVars

##Salvare 

class FASMachine():#(StateMachine):
    '''
    Finite State Machine implementation
    Composed by 4 states: ide, start, active and hold as in the original paper
    '''
    def __init__(self): 
        self.current_state = CommonVars.KALMAN_NULLSTATE
        self.previous_state = CommonVars.KALMAN_NULLSTATE
        self.innovation_threshold = CommonVars.INNOVATION_THRESHOLD

        self.distasq_threshold = CommonVars.START_DIST_THRESHOLD
        self.kalman_f = None
        self.T_bar = CommonVars.MAX_PRED_TIME
        self.next_time_pass_to = 'idle'
        self.last_valid_measurement = (np.zeros(2),0)
        self.previous_time = 'idle'
        self.needReset = False

    def debug(self):
        print('--------------------------------------------')
        print("CURRENT STATE: ",self.current_state)
        print("PREVIOUS STATE: ",self.previous_state)
        print("NEXT TIME PASS TO: ", self.next_time_pass_to)
        print("VALID MEASUREMENT: ", self.last_valid_measurement)
        print("PREVIOUS TIME: ", self.previous_time)

    def idle_state(self, rospy_time, measure):
        #print("need reset state:",self.needReset)
        if (self.needReset == True):
            self.needReset = False
            self.previous_time = 'idle'
            self.last_valid_measurement = ([0,0], rospy_time)
            self.current_state = CommonVars.KALMAN_NULLSTATE
            self.previous_state = CommonVars.KALMAN_NULLSTATE
            state_k = self.current_state

        if measure[0] != 0 or measure[1] != 0:
            state_k = np.array([measure[0], measure[1], 0, 0])
            self.next_time_pass_to = 'start'
        else:
            state_k = self.previous_state
            self.next_time_pass_to = 'idle'
        return state_k

    def start_state(self, rospy_time, measure):
        if measure[0] != 0 or measure[1] != 0:
            #distancesq = (self.previous_state[0] - measure[0])**2 + (self.previous_state[1] - measure[1])**2

            if (np.linalg.norm(measure-self.previous_state[0:2]) < self.distasq_threshold):
                delta_t = rospy_time - self.last_valid_measurement[1]
                print("deltat:", delta_t, "at:", rospy_time )
                state_k = np.array([measure[0],\
                                    measure[1], \
                                    (1/delta_t)*(measure[0] - self.previous_state[0]),\
                                    (1/delta_t)*(measure[1] - self.previous_state[1])])
                self.next_time_pass_to = 'active'
                self.kalman_f = Kalman(state_k, rospy_time, print_info = False) # We need to initialize the kalman filter
            else:
                print("MISMATCHED MEASURE! RESTARTING")
                self.next_time_pass_to = 'start'
                state_k = np.array([measure[0], measure[1], 0, 0])

        else:
            state_k = self.previous_state
            self.next_time_pass_to = 'idle'
            self.needReset = True
        return state_k


    def active_state(self, rospy_time, measure):      
        if measure[0] != 0 or measure[1] != 0:
            self.next_time_pass_to = 'active'
            self.kalman_f.predict(rospy_time)
            _ ,innovation = self.kalman_f.correct(measure[0],measure[1])
            if(np.linalg.norm(innovation) > self.innovation_threshold):
                print("RESET BY INNOVATION")
                print(innovation)
                #self.needReset = True
                #self.next_time_pass_to = 'idle'
                state_k = np.array([measure[0],measure[1],self.previous_state[2],self.previous_state[3]])
                self.next_time_pass_to = 'start'
            else:
                state_k = self.kalman_f.X_k_h
        else:
            state_k = self.previous_state
            self.next_time_pass_to = 'hold'
        return state_k

    def hold_state(self, rospy_time, measure):
        if measure[0] != 0 or measure[1] != 0:
            state_k = self.previous_state
            self.next_time_pass_to = 'active'
        else:
            if rospy_time <= (self.last_valid_measurement[1] + self.T_bar):
                self.kalman_f.predict(rospy_time, solo = True)
                state_k = self.kalman_f.X_k_h
                self.next_time_pass_to = 'hold'
            else:
                state_k = self.previous_state
                self.needReset = True
                self.next_time_pass_to = 'idle'
        return state_k 

    def propagate_state(self, rospy_time, state, N, DELTAT):
        state_arr = [CommonVars.KALMAN_NULLSTATE for i in range(0,N)]
        time = 0
        if self.kalman_f != None:
            for i in range(0,N):
                time = DELTAT * (i+1)
                state_arr[i] = self.kalman_f.predict_new_state(state,time)
        return state_arr




    def choose_state(self, rospy_time, measure, tiago_conf):
        self.previous_time = self.next_time_pass_to
        self.previous_state = np.copy(self.current_state)
        
        if self.next_time_pass_to == 'idle':
            self.current_state = self.idle_state(rospy_time, measure)
        elif self.next_time_pass_to == 'start':
            self.current_state = self.start_state(rospy_time, measure)

        elif self.next_time_pass_to == 'active':
            state_k = self.active_state(rospy_time, measure)
            self.current_state = state_k
            
        elif self.next_time_pass_to == 'hold':
            state_k = self.hold_state(rospy_time, measure)
            self.current_state = state_k
        
        if measure[0] != 0 or measure[1] != 0:
            self.last_valid_measurement = (measure, rospy_time)

        return self.current_state