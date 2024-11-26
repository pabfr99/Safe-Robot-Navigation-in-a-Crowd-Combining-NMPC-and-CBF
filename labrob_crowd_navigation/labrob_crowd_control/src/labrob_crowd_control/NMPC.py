import numpy as np
from numpy.linalg import *
import scipy.linalg

import casadi

from acados_template import AcadosModel, AcadosOcp, AcadosOcpConstraints, AcadosOcpCost, AcadosOcpOptions, AcadosOcpSolver

class NMPC:
    def __init__(self,
                 world_name: str = 'office', 
                 num_obstacles: int=15,
                 with_v_ref: bool = False):
        # Size of state and input:
        self.nq = 5
        self.nu = 2 

        self.num_obstacles = num_obstacles
        self.np = 4*self.num_obstacles
        # State indices:
        self.x_idx     = 0
        self.y_idx     = 1
        self.theta_idx = 2
        self.v_idx     = 3
        self.omega_idx = 4

        # Control input indices:
        self.r_wheel_idx = 0
        self.l_wheel_idx = 1

        # Kinematic parameters:
        self.wheel_radius = 0.0985 # [m]
        self.wheel_separation = 0.4044 # [m]
        self.b = 0.25 # [m]
        
        if with_v_ref:
            self.p_weight = 1e0 # position weights
            self.v_weight = 1e3 # driving velocity weight
            self.omega_weight = 1e0 # steering velocity weight
            self.u_weight = 1e0 # input weights
            self.terminal_factor = 1e2 # factor for the terminal state
            
        else:
            self.p_weight = 1e0 # position weights
            self.v_weight = 1e4 # driving velocity weight
            self.omega_weight = 1e0 # steering velocity weight
            self.u_weight = 1e0 # input weights
            self.terminal_factor = 5e3 # factor for the terminal state

        # Driving and steering velocity limits:
        self.driving_vel_max = 1.05 # [m/s]
        self.driving_vel_min = -0.25 # [m/s]
        self.steering_vel_max = 1.05 # [rad/s]
        self.steering_vel_min = -self.steering_vel_max

        # Driving and steering acceleration limits:
        self.driving_acc_max = 0.5 # [m/s^2]
        self.driving_acc_min = -self.driving_acc_max
        self.steering_acc_max = 1.05 # [rad/s^2]
        self.steering_acc_min = -self.steering_acc_max

        # Wheels velocity limits:
        self.w_max = self.driving_vel_max / self.wheel_radius # [rad/s], 10.1523
        self.w_min = -self.w_max

        # Wheels acceleration limits:
        self.alpha_max = self.driving_acc_max / self.wheel_radius # [rad/s^2], 5.0761
        self.alpha_min = -self.alpha_max

        # Number of control intervals:
        self.N = 50

        # Horizon duration:
        self.T = 1.0 # [s]
        
        # Add area constraints
        self.pred_motion = np.zeros((self.N, 2))

        self.define_areas(world_name)
        
        # Obstacle avoidance parameters
        self.dt = self.T / self.N
        self.rho=0.53/2.0
        self.d_s=0.4
        self.gamma=0.3
        
        # Crowd handler
        self.waiter = 5
        
        # Setup solver:
        self.acados_ocp_solver = self.__create_acados_ocp_solver(self.N, self.T)

    def define_areas(self, world_name):
        if world_name == 'office':
            self.vertex_num = 6
            self.areas = np.array([
                [[4.7, -4.7], [4.7, 4.7], [-4.7, 4.7], [-4.7, -4.7], [np.nan, np.nan], [np.nan, np.nan]],
                [[1, -4], [2, -4], [2, 9], [1, 9], [np.nan, np.nan], [np.nan, np.nan]],
                [[4.7, 5.3], [4.7, 9.7], [-4.7, 9.7], [-4.7, 5.3], [np.nan, np.nan], [np.nan, np.nan]]
            ])
            
        elif world_name=='double':
            self.vertex_num = 6
            self.areas = np.array([
                [[0, -3.5], [0, -0.3], [-4.7, -0.3], [-4.7, -4.7], [-3,-4.7], [np.nan, np.nan]],
                [[4.7, -4.7], [4.7, -2.5], [-3.0, -2.5], [0.0, -4.7], [np.nan, np.nan], [np.nan, np.nan]],
                [[4.7, -4.7], [4.7, -0.2], [1.5, -0.2], [2.5, -4.7], [np.nan, np.nan], [np.nan, np.nan]],
                [[3, -2.0], [3, 4.7], [2, 4.7], [2, -2.0], [np.nan, np.nan], [np.nan, np.nan]],
                [[4.7, 0.5], [4.7, 3.0], [-1.0, 4.7], [-4.7, 4.7], [-4.7, 0.5], [np.nan, np.nan]],
                [[-2.0, 0.5], [-2.0, 9.7], [-3.0, 9.7], [-3.0, 0.5], [np.nan, np.nan], [np.nan, np.nan]],
                [[4.7, 5.5], [4.7, 9.5], [-0.5, 9.5], [-4.7, 6.5], [-4.7, 5.5], [np.nan, np.nan]]
            ])
        elif world_name=='corridor':
            self.vertex_num = 6
            self.areas = np.array([
                [[-2.0, 0.0], [-9.0, 0.0], [-9.0, -2.5], [-4.0, -4.7], [-2.0, -4.7], [np.nan, np.nan]],
                [[-8.7, -4.0], [-6.2, -4.0], [-6.2, 4.7], [-7.7, 4.7], [-8.7, 3.7], [np.nan, np.nan]],
                [[-7.7, 0.2], [-5.2, 0.2], [-5.2, 9.7], [-6.2, 9.7], [-7.7, 6.0], [np.nan, np.nan]],
                [[-4.0, 5.2], [-0.2, 7.0], [-0.2, 8.2], [-3.0, 9.7], [-7.0, 9.7], [-9.7, 5.2]]
            ])
        else:
            self.vertex_num = 6
            self.areas = np.array([
                [[4.7, -4.7], [4.7, 4.7], [-4.7, 4.7], [-4.7, -4.7], [np.nan, np.nan], [np.nan, np.nan]],
                [[1, -4], [2, -4], [2, 9], [1, 9], [np.nan, np.nan], [np.nan, np.nan]],
                [[4.7, 5.3], [4.7, 9.7], [-4.7, 9.7], [-4.7, 5.3], [np.nan, np.nan], [np.nan, np.nan]]
            ])

    def init(self, q0):
        for k in range(self.N):
            self.acados_ocp_solver.set(k, 'x', q0)
            self.acados_ocp_solver.set(k, 'u', np.zeros(self.nu))
            
        self.acados_ocp_solver.set(self.N, 'x', q0)

    # Nonlinear constraint functions:
    def RK4(self, f, x0, u ,dt):
        k1 = f(x0, u)
        k2 = f(x0 + k1 * dt / 2.0, u)
        k3 = f(x0 + k2 * dt / 2.0, u)
        k4 = f(x0 + k3 * dt, u)
        yf = x0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return yf

    # Systems dynamics:
    def __f(self, x, u):
        xdot = casadi.SX.zeros(self.nq)
        xdot[self.x_idx] = self.__x_dot(x)
        xdot[self.y_idx] = self.__y_dot(x)
        xdot[self.theta_idx] = self.__theta_dot(x)
        xdot[self.v_idx] = self.__v_dot(u)
        xdot[self.omega_idx] = self.__omega_dot(u)
        return xdot

    def __x_dot(self, q):
        b = self.b
        theta = q[self.theta_idx]
        v = q[self.v_idx]
        omega = q[self.omega_idx]
        return v * casadi.cos(theta) - omega * b * casadi.sin(theta)

    def __y_dot(self, q):
        b = self.b
        theta = q[self.theta_idx]
        v = q[self.v_idx]
        omega = q[self.omega_idx]
        return v * casadi.sin(theta) + omega * b * casadi.cos(theta)
    
    def __theta_dot(self, q):
        return q[self.omega_idx]
    
    def __v_dot(self, u):
        alpha_r = u[self.r_wheel_idx]
        alpha_l = u[self.l_wheel_idx]
        wheel_radius = self.wheel_radius
        return wheel_radius * 0.5 * (alpha_r + alpha_l)
    
    def __omega_dot(self, u):
        alpha_r = u[self.r_wheel_idx]
        alpha_l = u[self.l_wheel_idx]
        wheel_radius = self.wheel_radius
        wheel_separation = self.wheel_separation
        return (wheel_radius / wheel_separation) * (alpha_r - alpha_l)
    
    def __create_acados_model(self) -> AcadosModel:
        # Setup CasADi expressions:
        q = casadi.SX.sym('q', self.nq)
        qdot = casadi.SX.sym('qdot', self.nq)
        u = casadi.SX.sym('u', self.nu)
        p = casadi.SX.sym('p', self.np)
        f_expl = self.__f(q, u)
        f_impl = qdot - f_expl


        # Create acados model:
        acados_model = AcadosModel()
        acados_model.name = 'tiago_kinematic_model'

        # System dynamics:
        acados_model.f_impl_expr = f_impl
        acados_model.f_expl_expr = f_expl

        # Variables and params:
        acados_model.x = q
        acados_model.xdot = qdot
        acados_model.u = u
        acados_model.p = p
        acados_model.con_h_expr = self.cbf_constraint(q, u, p)

        return acados_model
    
    def __create_acados_cost(self) -> AcadosOcpCost:
        acados_cost = AcadosOcpCost()

        # Set wheighting matrices
        Q_mat = np.diag([self.p_weight, self.p_weight, 0.0]) # [x, y, theta]
        R_mat = np.diag([self.v_weight, self.omega_weight]) # [v, omega]
        S_mat = np.diag([self.u_weight, self.u_weight]) # [alphar, alphal]

        acados_cost.cost_type   = 'LINEAR_LS'
        acados_cost.cost_type_e = 'LINEAR_LS'
        
        ny = self.nq + self.nu
        ny_e = self.nq

        acados_cost.W_e = scipy.linalg.block_diag(self.terminal_factor * Q_mat, R_mat)
        acados_cost.W = scipy.linalg.block_diag(Q_mat, R_mat, S_mat)

        Vx = np.zeros((ny, self.nq))
        Vx[:self.nq, :self.nq] = np.eye(self.nq)
        acados_cost.Vx = Vx

        Vu = np.zeros((ny, self.nu))
        Vu[self.nq:ny, 0:self.nu] = np.eye(self.nu)
        acados_cost.Vu = Vu

        acados_cost.Vx_e = np.eye(ny_e)

        acados_cost.yref = np.zeros((ny,))
        acados_cost.yref_e = np.zeros((ny_e,))
        
        return acados_cost
    
    
    def __create_acados_constraints(self) -> AcadosOcpConstraints:

        acados_constraints = AcadosOcpConstraints()

        # Linear inequality constraints on the state:
        acados_constraints.idxbx = np.array([self.v_idx, self.omega_idx])
        acados_constraints.lbx = np.array([self.driving_vel_min, self.steering_vel_min])
        acados_constraints.ubx = np.array([self.driving_vel_max, self.steering_vel_max])
        acados_constraints.x0 = np.zeros(self.nq)

        # Linear inequality constraints on the inputs:
        acados_constraints.idxbu = np.array([self.r_wheel_idx, self.l_wheel_idx])
        acados_constraints.lbu = np.array([self.alpha_min, self.alpha_min])
        acados_constraints.ubu = np.array([self.alpha_max, self.alpha_max])

        # Linear constraints on wheel velocities and driving/steering acceleration
        # expressed in terms of state and input
        C_mat = np.zeros((4+self.vertex_num, self.nq))
        C_mat[:2, 3] = (1.0 / self.wheel_radius)
        C_mat[:2, 4] = self.wheel_separation / (2.0 * self.wheel_radius) * np.array([1.0, -1.0])

        a, b, c = self.get_constraint_coefficients(0)
        C_mat[4:, 0] = a
        C_mat[4:, 1] = b

        D_mat = np.zeros((4+self.vertex_num, self.nu))
        D_mat[2, :] = self.wheel_radius * 0.5
        D_mat[3, :] = (self.wheel_radius / self.wheel_separation) * np.array([1.0, -1.0])
        
        acados_constraints.D = D_mat
        acados_constraints.C = C_mat
        
        acados_constraints.lg = np.concatenate((np.array([self.w_min,
                                                          self.w_min,
                                                          self.driving_acc_min,
                                                          self.steering_acc_min]),np.full(self.vertex_num, -1000)))
        acados_constraints.ug = np.concatenate((np.array([self.w_max,
                                                          self.w_max,
                                                          self.driving_acc_max,
                                                          self.steering_acc_max]),c))
        acados_constraints.lh = np.zeros(self.num_obstacles) 
        acados_constraints.uh = 1000 * np.ones(self.num_obstacles)

        return acados_constraints
    
    def __create_acados_solver_options(self, T) -> AcadosOcpOptions:
        acados_solver_options = AcadosOcpOptions()
        acados_solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        acados_solver_options.hpipm_mode = 'SPEED'
        acados_solver_options.hessian_approx = 'GAUSS_NEWTON'
        acados_solver_options.integrator_type = 'ERK'
        acados_solver_options.print_level = 0
        acados_solver_options.nlp_solver_type = 'SQP_RTI'
        acados_solver_options.tf = T

        return acados_solver_options
    
    def __create_acados_ocp(self, N, T) -> AcadosOcp:
        acados_ocp = AcadosOcp()
        acados_ocp.model = self.__create_acados_model()
        acados_ocp.dims.N = N
        acados_ocp.cost = self.__create_acados_cost()
        acados_ocp.constraints = self.__create_acados_constraints()
        acados_ocp.solver_options = self.__create_acados_solver_options(T)
        
        return acados_ocp
    
    def __create_acados_ocp_solver(self, N, T, use_cython=False) -> AcadosOcpSolver:
        acados_ocp = self.__create_acados_ocp(N, T)
        acados_ocp.parameter_values = np.zeros(self.np)
        if use_cython:
            AcadosOcpSolver.generate(acados_ocp, json_file='acados_ocp_nlp.json')
            AcadosOcpSolver.build(acados_ocp.code_export_directory, with_cython=True)
            return AcadosOcpSolver.create_cython_solver('acados_ocp_nlp.json')
        else:
            return AcadosOcpSolver(acados_ocp)
        
    def compute_cbf(self, 
                    robot_state,  
                    obstacle_position):
        robotx = robot_state[self.x_idx]
        roboty = robot_state[self.y_idx]
        c = casadi.vertcat(robotx, roboty)
        distance = casadi.norm_2(c-obstacle_position)
        h = distance**2 - (self.rho+self.d_s)**2
        return h

    def cbf_constraint(self, x, u, p):
        h_constraints = []
        for i in range(self.num_obstacles):
            obstacle_pos_current = p[4*i:4*i+2]
            obstacle_pos_next = p[4*i+2:4*i+4]
            robot_state_next = self.RK4(self.__f, x, u, self.dt)
            h_current = self.compute_cbf(x, obstacle_pos_current)
            h_next = self.compute_cbf(robot_state_next, obstacle_pos_next)
            delta_h = h_next - h_current
            cbf_condition = delta_h + self.gamma * h_current
            h_constraints.append(cbf_condition)
        return casadi.vertcat(*h_constraints)


    def get_constraint_coefficients(self,position_constraint_index):
        area=self.areas[position_constraint_index]
        index=np.where(np.isnan(area))
        index=index[0][0] if (len(index[0])>0 and index[0][0]!= None) else self.vertex_num+1
        shifted_area=np.append(area[1:index],[area[0]],axis=0)
        a=np.zeros(self.vertex_num)
        b=np.zeros(self.vertex_num)
        c=np.ones(self.vertex_num)*1000
        a[:index]=shifted_area[:index,1]-area[:index,1]
        b[:index]=area[:index,0]-shifted_area[:index,0]
        c[:index]=np.multiply(area[:index,0],shifted_area[:index,1])-np.multiply(area[:index,1],shifted_area[:index,0])
        return a,b,c
        
    def predict_obstacle_trajectory(self, obstacle_pos, obstacle_vel):
        # Time step for each prediction
        dt = self.T / self.N  

        predicted_positions = np.zeros((self.N+1, 2))

        x, y = obstacle_pos.x, obstacle_pos.y
        vx, vy = obstacle_vel.x, obstacle_vel.y
        for i in range(self.N+1):
            x += vx * dt
            y += vy * dt
            predicted_positions[i] = [x, y]

        return predicted_positions
    
    def predict_obstacles_horizon(self, crowd_prediction):

        predicted_obstacle_trajectories = []
        if self.waiter < 0:
            for obstacle in crowd_prediction:
                predicted_obstacle_trajectories.append(self.predict_obstacle_trajectory(obstacle.position, obstacle.velocity))
        else:
            self.waiter -=1
        return predicted_obstacle_trajectories

    def update(
            self,
            q0: np.array,
            q_ref: np.array,
            u_ref: np.array,
            position_constraint_index, 
            crowd):
        predicted_obstacle_trajectories = self.predict_obstacles_horizon(crowd)

        # Set parameters
        for k in range(self.N):
            self.acados_ocp_solver.set(k, 'y_ref', np.concatenate((q_ref[:, k], u_ref[:, k])))
    
            a, b, c = self.get_constraint_coefficients(position_constraint_index[k])
            C_mat = self.acados_ocp_solver.acados_ocp.constraints.C
            C_mat[4:, 0] = a
            C_mat[4:, 1] = b

            ug_vec = self.acados_ocp_solver.acados_ocp.constraints.ug
            ug_vec[4:] = c

            self.acados_ocp_solver.constraints_set(k, 'C', C_mat, api='new')
            self.acados_ocp_solver.constraints_set(k, 'ug', ug_vec)
            

            # Set obstacle positions
            pi = []
            for obstacle_trajectory in predicted_obstacle_trajectories:
                obstacle_position_current = obstacle_trajectory[k] 
                obstacle_position_next = obstacle_trajectory[k+1]
                pi.extend(obstacle_position_current)
                pi.extend(obstacle_position_next)
            if len(pi) < self.np:
                pi.extend(10*np.ones(self.np - len(pi)))
            self.acados_ocp_solver.set(k, 'p', np.array(pi))

        self.acados_ocp_solver.set(self.N, 'y_ref', q_ref[:, self.N])

        # Solve NLP
        self.u0 = self.acados_ocp_solver.solve_for_x0(q0)
        for k in range(self.N):
            self.pred_motion[k] = self.acados_ocp_solver.get(k, 'x')[:2]

    def get_command(self):
        return self.u0
    
    def get_predicted_motion(self):
        return self.pred_motion