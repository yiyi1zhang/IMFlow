# https://wilselby.com/research/arducopter/modeling/
import numpy as np
import math
import scipy.signal as scisi
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import sys
sys.path.append(r"..")

import os
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3
import vehiclemodels.utils.tire_model as tireModel
from vehiclemodels.init_std import init_std
from vehiclemodels.init_mb import init_mb
from vehiclemodels.utils.acceleration_constraints import acceleration_constraints
from vehiclemodels.utils.steering_constraints import steering_constraints
from vehiclemodels.utils.vehicle_dynamics_ks_cog import vehicle_dynamics_ks_cog

vehicle = 3

def kST_fun(x, t, u, time_points):
    # states
    # x1 = x-position in a global coordinate system
    # x2 = y-position in a global coordinate system
    # x3 = velocity in x-direction
    # x4 = yaw angle

    # u1 = steering angle front wheels
    # u2 = longitudinal acceleration
    u1, u2 = u
    interp1 = interp1d(time_points, u1, axis=0, kind='nearest')
    interp2 = interp1d(time_points, u2, axis=0, kind='nearest')
    u1 = interp1(t)
    u2 = interp2(t)
    # if u1 < p.steering.min:
    #     u1 = p.steering.min
    # elif u1 > p.steering.max:
    #     u1 = p.steering.max
    # u2 = acceleration_constraints(x[2], u2, p.longitudinal)
    # system dynamics
    f = [x[2] * math.cos(x[3]),
         x[2] * math.sin(x[3]),
         u2,
         x[2] * math.tan(u1) / 0.3]

    return f


def ST_fun(x, t, p, u, time_points):
    # states
    # x1 = x-position in a global coordinate system
    # x2 = y-position in a global coordinate system
    # x3 = steering angle of front wheels
    # x4 = velocity in x-direction
    # x5 = yaw angle
    # x6 = yaw rate
    # x7 = slip angle at vehicle center

    # u1 = steering angle velocity of front wheels
    # u2 = longitudinal acceleration

    u1, u2 = u
    u1 = np.interp(t, time_points, u1)
    u2 = np.interp(t, time_points, u2)

    # create equivalent bicycle parameters

    C_Sf = -p.tire.p_ky1 / p.tire.p_dy1
    C_Sr = -p.tire.p_ky1 / p.tire.p_dy1

    # consider steering constraints
    u = []
    u.append(steering_constraints(x[2], u1, p.steering))  # different name u_init/u due to side effects of u
    # consider acceleration constraints
    u.append(
        acceleration_constraints(x[3], u2, p.longitudinal))  # different name u_init/u due to side effects of u

    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        # Use kinematic model with reference point at center of mass
        # wheelbase
        lwb = p.a + p.b
        # system dynamics
        x_ks = [x[0], x[1], x[2], x[3], x[4]]
        # kinematic model
        f_ks = vehicle_dynamics_ks_cog(x_ks, u, p)
        f = [f_ks[0], f_ks[1], f_ks[2], f_ks[3], f_ks[4]]
        # derivative of slip angle and yaw rate
        d_beta = (p.b * u[0]) / (lwb * math.cos(x[2]) ** 2 * (1 + (math.tan(x[2]) ** 2 * p.b / lwb) ** 2))
        dd_psi = 1 / lwb * (u[1] * math.cos(x[6]) * math.tan(x[2]) -
                            x[3] * math.sin(x[6]) * d_beta * math.tan(x[2]) +
                            x[3] * math.cos(x[6]) * u[0] / math.cos(x[2]) ** 2)
        f.append(dd_psi)
        f.append(d_beta)

    else:
        # system dynamics
        f = [x[3] * math.cos(x[6] + x[4]),
             x[3] * math.sin(x[6] + x[4]),
             u[0],
             u[1],
             x[5],
             - p.mu * p.m / (x[3] * p.I_z * (p.l_r + p.l_f)) * (p.l_f ** 2 * C_Sf * (p.g * p.l_r - u[1] * p.h_s) +
                                                                p.l_r ** 2 * C_Sr * (p.g * p.l_f + u[1] * p.h_s)) * x[
                 5] +
             p.mu * p.m / (p.I_z * (p.l_r + p.l_f)) * (p.l_r * C_Sr * (p.g * p.l_f + u[1] * p.h_s) -
                                                       p.l_f * C_Sf * (p.g * p.l_r - u[1] * p.h_s)) * x[6] +
             p.mu * p.m / (p.I_z * (p.l_r + p.l_f)) * p.l_f * C_Sf * (p.g * p.l_r - u[1] * p.h_s) * x[2],
             (p.mu / (x[3] ** 2 * (p.l_r + p.l_f)) * (C_Sr * (p.g * p.l_f + u[1] * p.h_s) * p.l_r -
                                                      C_Sf * (p.g * p.l_r - u[1] * p.h_s) * p.l_f) - 1) * x[5] -
             p.mu / (x[3] * (p.l_r + p.l_f)) * (C_Sr * (p.g * p.l_f + u[1] * p.h_s) +
                                                C_Sf * (p.g * p.l_r - u[1] * p.h_s)) * x[6]
             + p.mu / (x[3] * (p.l_r + p.l_f)) * (C_Sf * (p.g * p.l_r - u[1] * p.h_s)) * x[2]]

    return f


def STD_fun(x, t, p, u, time_points):
    # states
    # x1 = x-position in a global coordinate system
    # x2 = y-position in a global coordinate system
    # x3 = steering angle of front wheels
    # x4 = velocity at vehicle center
    # x5 = yaw angle
    # x6 = yaw rate
    # x7 = slip angle at vehicle center
    # x8 = front wheel angular speed
    # x9 = rear wheel angular speed

    # u1 = steering angle velocity of front wheels
    # u2 = longitudinal acceleration
    tStart = 0  # start time
    tFinal = 3  # final time
    tStep = 1e-2  # step size
    seq_len = int((tFinal - tStart) / tStep) + 1

    u1, u2 = u
    u1 = np.interp(t, time_points, u1)
    u2 = np.interp(t, time_points, u2)

    # steering and acceleration constraints
    u = []
    u.append(steering_constraints(x[2], u1, p.steering))  # different name due to side effects of u
    u.append(acceleration_constraints(x[3], u2, p.longitudinal))  # different name due to side effect of u

    # compute lateral tire slip angles
    alpha_f = math.atan((x[3] * math.sin(x[6]) + x[5] * p.l_f) / (x[3] * math.cos(x[6]))) - x[2] if x[
                                                                                                        3] > p.v_min else 0
    alpha_r = math.atan((x[3] * math.sin(x[6]) - x[5] * p.l_r) / (x[3] * math.cos(x[6]))) if x[3] > p.v_min else 0

    # compute vertical tire forces
    F_zf = p.m * (-u[1] * p.h_s + p.g * p.l_r) / (p.l_r + p.l_f)
    F_zr = p.m * (u[1] * p.h_s + p.g * p.l_f) / (p.l_r + p.l_f)

    # compute front and rear tire speeds
    u_wf = max(0, x[3] * math.cos(x[6]) * math.cos(x[2]) + (x[3] * math.sin(x[6]) + p.a * x[5]) * math.sin(x[2]))
    u_wr = max(0, x[3] * math.cos(x[6]))

    # compute longitudinal tire slip
    s_f = 1 - p.R_w * x[7] / max(u_wf, p.v_min)
    s_r = 1 - p.R_w * x[8] / max(u_wr, p.v_min)

    # compute tire forces (Pacejka)
    # pure slip longitudinal forces
    F0_xf = tireModel.formula_longitudinal(s_f, 0, F_zf, p.tire)
    F0_xr = tireModel.formula_longitudinal(s_r, 0, F_zr, p.tire)

    # pure slip lateral forces
    res = tireModel.formula_lateral(alpha_f, 0, F_zf, p.tire)
    F0_yf = res[0]
    mu_yf = res[1]
    res = tireModel.formula_lateral(alpha_r, 0, F_zr, p.tire)
    F0_yr = res[0]
    mu_yr = res[1]

    # combined slip longitudinal forces
    F_xf = tireModel.formula_longitudinal_comb(s_f, alpha_f, F0_xf, p.tire)
    F_xr = tireModel.formula_longitudinal_comb(s_r, alpha_r, F0_xr, p.tire)

    # combined slip lateral forces
    F_yf = tireModel.formula_lateral_comb(s_f, alpha_f, 0, mu_yf, F_zf, F0_yf, p.tire)
    F_yr = tireModel.formula_lateral_comb(s_r, alpha_r, 0, mu_yr, F_zr, F0_yr, p.tire)

    # convert acceleration input to brake and engine torque
    if u[1] > 0:
        T_B = 0.0
        T_E = p.m * p.R_w * u[1]
    else:
        T_B = p.m * p.R_w * u[1]
        T_E = 0.

    # system dynamics
    d_v = 1 / p.m * (-F_yf * math.sin(x[2] - x[6]) + F_yr * math.sin(x[6]) + F_xr * math.cos(x[6]) + F_xf * math.cos(
        x[2] - x[6]))
    dd_psi = 1 / p.I_z * (F_yf * math.cos(x[2]) * p.l_f - F_yr * p.l_r + F_xf * math.sin(x[2]) * p.l_f)
    d_beta = -x[5] + 1 / (p.m * x[3]) * (
            F_yf * math.cos(x[2] - x[6]) + F_yr * math.cos(x[6]) - F_xr * math.sin(x[6]) + F_xf * math.sin(
        x[2] - x[6])) if x[3] > p.v_min else 0

    # wheel dynamics (negative wheel spin forbidden)
    d_omega_f = 1 / p.I_y_w * (-p.R_w * F_xf + p.T_sb * T_B + p.T_se * T_E) if x[7] >= 0 else 0
    x[7] = max(0, x[7])
    d_omega_r = 1 / p.I_y_w * (-p.R_w * F_xr + (1 - p.T_sb) * T_B + (1 - p.T_se) * T_E) if x[8] >= 0 else 0
    x[8] = max(0, x[8])

    # *** Mix with kinematic model at low speeds ***
    # kinematic system dynamics
    x_ks = [x[0], x[1], x[2], x[3], x[4]]
    f_ks = vehicle_dynamics_ks_cog(x_ks, u, p)
    # derivative of slip angle and yaw rate (kinematic)
    d_beta_ks = (p.b * u[0]) / (p.l_wb * math.cos(x[2]) ** 2 * (1 + (math.tan(x[2]) ** 2 * p.b / p.l_wb) ** 2))
    dd_psi_ks = 1 / p.l_wb * (u[1] * math.cos(x[6]) * math.tan(x[2]) -
                              x[3] * math.sin(x[6]) * d_beta_ks * math.tan(x[2]) +
                              x[3] * math.cos(x[6]) * u[0] / math.cos(x[2]) ** 2)
    # derivative of angular speeds (kinematic)
    d_omega_f_ks = (1 / 0.02) * (u_wf / p.R_w - x[7])
    d_omega_r_ks = (1 / 0.02) * (u_wr / p.R_w - x[8])

    # weights for mixing both models
    w_std = 0.5 * (math.tanh((x[3] - p.v_s) / p.v_b) + 1)
    w_ks = 1 - w_std

    # output vector: mix results of dynamic and kinematic model
    f = [x[3] * math.cos(x[6] + x[4]),
         x[3] * math.sin(x[6] + x[4]),
         u[0],
         w_std * d_v + w_ks * f_ks[3],
         w_std * x[5] + w_ks * f_ks[4],
         w_std * dd_psi + w_ks * dd_psi_ks,
         w_std * d_beta + w_ks * d_beta_ks,
         w_std * d_omega_f + w_ks * d_omega_f_ks,
         w_std * d_omega_r + w_ks * d_omega_r_ks]

    return f


def MB_fun(x, t, p, u, time_points):
    # states
    # x1 = x-position in a global coordinate system
    # x2 = y-position in a global coordinate system
    # x3 = steering angle of front wheels
    # x4 = velocity in x-direction
    # x5 = yaw angle
    # x6 = yaw rate

    # x7 = roll angle
    # x8 = roll rate
    # x9 = pitch angle
    # x10 = pitch rate
    # x11 = velocity in y-direction
    # x12 = z-position
    # x13 = velocity in z-direction

    # x14 = roll angle front
    # x15 = roll rate front
    # x16 = velocity in y-direction front
    # x17 = z-position front
    # x18 = velocity in z-direction front

    # x19 = roll angle rear
    # x20 = roll rate rear
    # x21 = velocity in y-direction rear
    # x22 = z-position rear
    # x23 = velocity in z-direction rear

    # x24 = left front wheel angular speed
    # x25 = right front wheel angular speed
    # x26 = left rear wheel angular speed
    # x27 = right rear wheel angular speed

    # x28 = delta_y_f
    # x29 = delta_y_r

    # u1 = steering angle velocity of front wheels
    # u2 = acceleration
    tStart = 0  # start time
    tFinal = 3  # final time
    tStep = 1e-2  # step size
    seq_len = int((tFinal - tStart) / tStep) + 1

    u1, u2 = u
    u1 = np.interp(t, time_points, u1)
    u2 = np.interp(t, time_points, u2)

    u = []
    u.append(steering_constraints(x[2], u1, p.steering))  # different name u_init/u due to side effects of u
    # consider acceleration constraints
    u.append(
        acceleration_constraints(x[3], u2, p.longitudinal))  # different name u_init/u due to side effects of u

    # compute slip angle at cg
    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        beta = 0.
    else:
        beta = math.atan(x[10] / x[3])
    vel = math.sqrt(x[3] ** 2 + x[10] ** 2)

    # vertical tire forces
    F_z_LF = (x[16] + p.R_w * (math.cos(x[13]) - 1) - 0.5 * p.T_f * math.sin(x[13])) * p.K_zt
    F_z_RF = (x[16] + p.R_w * (math.cos(x[13]) - 1) + 0.5 * p.T_f * math.sin(x[13])) * p.K_zt
    F_z_LR = (x[21] + p.R_w * (math.cos(x[18]) - 1) - 0.5 * p.T_r * math.sin(x[18])) * p.K_zt
    F_z_RR = (x[21] + p.R_w * (math.cos(x[18]) - 1) + 0.5 * p.T_r * math.sin(x[18])) * p.K_zt

    # obtain individual tire speeds
    u_w_lf = (x[3] + 0.5 * p.T_f * x[5]) * math.cos(x[2]) + (x[10] + p.a * x[5]) * math.sin(x[2])
    u_w_rf = (x[3] - 0.5 * p.T_f * x[5]) * math.cos(x[2]) + (x[10] + p.a * x[5]) * math.sin(x[2])
    u_w_lr = x[3] + 0.5 * p.T_r * x[5]
    u_w_rr = x[3] - 0.5 * p.T_r * x[5]

    # negative wheel spin forbidden
    if u_w_lf < 0.0:
        u_w_lf *= 0

    if u_w_rf < 0.0:
        u_w_rf *= 0

    if u_w_lr < 0.0:
        u_w_lr *= 0

    if u_w_rr < 0.0:
        u_w_rr *= 0
    # compute longitudinal slip
    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        s_lf = 0.
        s_rf = 0.
        s_lr = 0.
        s_rr = 0.
    else:
        s_lf = 1 - p.R_w * x[23] / u_w_lf
        s_rf = 1 - p.R_w * x[24] / u_w_rf
        s_lr = 1 - p.R_w * x[25] / u_w_lr
        s_rr = 1 - p.R_w * x[26] / u_w_rr

        # lateral slip angles
    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        alpha_LF = 0.
        alpha_RF = 0.
        alpha_LR = 0.
        alpha_RR = 0.
    else:
        alpha_LF = math.atan((x[10] + p.a * x[5] - x[14] * (p.R_w - x[16])) / (x[3] + 0.5 * p.T_f * x[5])) - x[2]
        alpha_RF = math.atan((x[10] + p.a * x[5] - x[14] * (p.R_w - x[16])) / (x[3] - 0.5 * p.T_f * x[5])) - x[2]
        alpha_LR = math.atan((x[10] - p.b * x[5] - x[19] * (p.R_w - x[21])) / (x[3] + 0.5 * p.T_r * x[5]))
        alpha_RR = math.atan((x[10] - p.b * x[5] - x[19] * (p.R_w - x[21])) / (x[3] - 0.5 * p.T_r * x[5]))

        # auxiliary suspension movement
    z_SLF = (p.h_s - p.R_w + x[16] - x[11]) / math.cos(x[6]) - p.h_s + p.R_w + p.a * x[8] + 0.5 * (x[6] - x[13]) * p.T_f
    z_SRF = (p.h_s - p.R_w + x[16] - x[11]) / math.cos(x[6]) - p.h_s + p.R_w + p.a * x[8] - 0.5 * (x[6] - x[13]) * p.T_f
    z_SLR = (p.h_s - p.R_w + x[21] - x[11]) / math.cos(x[6]) - p.h_s + p.R_w - p.b * x[8] + 0.5 * (x[6] - x[18]) * p.T_r
    z_SRR = (p.h_s - p.R_w + x[21] - x[11]) / math.cos(x[6]) - p.h_s + p.R_w - p.b * x[8] - 0.5 * (x[6] - x[18]) * p.T_r

    dz_SLF = x[17] - x[12] + p.a * x[9] + 0.5 * (x[7] - x[14]) * p.T_f
    dz_SRF = x[17] - x[12] + p.a * x[9] - 0.5 * (x[7] - x[14]) * p.T_f
    dz_SLR = x[22] - x[12] - p.b * x[9] + 0.5 * (x[7] - x[19]) * p.T_r
    dz_SRR = x[22] - x[12] - p.b * x[9] - 0.5 * (x[7] - x[19]) * p.T_r

    # camber angles
    gamma_LF = x[6] + p.D_f * z_SLF + p.E_f * (z_SLF) ** 2
    gamma_RF = x[6] - p.D_f * z_SRF - p.E_f * (z_SRF) ** 2
    gamma_LR = x[6] + p.D_r * z_SLR + p.E_r * (z_SLR) ** 2
    gamma_RR = x[6] - p.D_r * z_SRR - p.E_r * (z_SRR) ** 2

    # compute longitudinal tire forces using the magic formula for pure slip
    F0_x_LF = tireModel.formula_longitudinal(s_lf, gamma_LF, F_z_LF, p.tire)
    F0_x_RF = tireModel.formula_longitudinal(s_rf, gamma_RF, F_z_RF, p.tire)
    F0_x_LR = tireModel.formula_longitudinal(s_lr, gamma_LR, F_z_LR, p.tire)
    F0_x_RR = tireModel.formula_longitudinal(s_rr, gamma_RR, F_z_RR, p.tire)

    # compute lateral tire forces using the magic formula for pure slip
    res = tireModel.formula_lateral(alpha_LF, gamma_LF, F_z_LF, p.tire)
    F0_y_LF = res[0]
    mu_y_LF = res[1]
    res = tireModel.formula_lateral(alpha_RF, gamma_RF, F_z_RF, p.tire)
    F0_y_RF = res[0]
    mu_y_RF = res[1]
    res = tireModel.formula_lateral(alpha_LR, gamma_LR, F_z_LR, p.tire)
    F0_y_LR = res[0]
    mu_y_LR = res[1]
    res = tireModel.formula_lateral(alpha_RR, gamma_RR, F_z_RR, p.tire)
    F0_y_RR = res[0]
    mu_y_RR = res[1]

    # compute longitudinal tire forces using the magic formula for combined slip
    F_x_LF = tireModel.formula_longitudinal_comb(s_lf, alpha_LF, F0_x_LF, p.tire)
    F_x_RF = tireModel.formula_longitudinal_comb(s_rf, alpha_RF, F0_x_RF, p.tire)
    F_x_LR = tireModel.formula_longitudinal_comb(s_lr, alpha_LR, F0_x_LR, p.tire)
    F_x_RR = tireModel.formula_longitudinal_comb(s_rr, alpha_RR, F0_x_RR, p.tire)

    # compute lateral tire forces using the magic formula for combined slip
    F_y_LF = tireModel.formula_lateral_comb(s_lf, alpha_LF, gamma_LF, mu_y_LF, F_z_LF, F0_y_LF, p.tire)
    F_y_RF = tireModel.formula_lateral_comb(s_rf, alpha_RF, gamma_RF, mu_y_RF, F_z_RF, F0_y_RF, p.tire)
    F_y_LR = tireModel.formula_lateral_comb(s_lr, alpha_LR, gamma_LR, mu_y_LR, F_z_LR, F0_y_LR, p.tire)
    F_y_RR = tireModel.formula_lateral_comb(s_rr, alpha_RR, gamma_RR, mu_y_RR, F_z_RR, F0_y_RR, p.tire)

    # auxiliary movements for compliant joint equations
    delta_z_f = p.h_s - p.R_w + x[16] - x[11]
    delta_z_r = p.h_s - p.R_w + x[21] - x[11]

    delta_phi_f = x[6] - x[13]
    delta_phi_r = x[6] - x[18]

    dot_delta_phi_f = x[7] - x[14]
    dot_delta_phi_r = x[7] - x[19]

    dot_delta_z_f = x[17] - x[12]
    dot_delta_z_r = x[22] - x[12]

    dot_delta_y_f = x[10] + p.a * x[5] - x[15]
    dot_delta_y_r = x[10] - p.b * x[5] - x[20]

    delta_f = delta_z_f * math.sin(x[6]) - x[27] * math.cos(x[6]) - (p.h_raf - p.R_w) * math.sin(delta_phi_f)
    delta_r = delta_z_r * math.sin(x[6]) - x[28] * math.cos(x[6]) - (p.h_rar - p.R_w) * math.sin(delta_phi_r)

    dot_delta_f = (delta_z_f * math.cos(x[6]) + x[27] * math.sin(x[6])) * x[7] + dot_delta_z_f * math.sin(
        x[6]) - dot_delta_y_f * math.cos(x[6]) - (p.h_raf - p.R_w) * math.cos(delta_phi_f) * dot_delta_phi_f
    dot_delta_r = (delta_z_r * math.cos(x[6]) + x[28] * math.sin(x[6])) * x[7] + dot_delta_z_r * math.sin(
        x[6]) - dot_delta_y_r * math.cos(x[6]) - (p.h_rar - p.R_w) * math.cos(delta_phi_r) * dot_delta_phi_r

    # compliant joint forces
    F_RAF = delta_f * p.K_ras + dot_delta_f * p.K_rad
    F_RAR = delta_r * p.K_ras + dot_delta_r * p.K_rad

    # auxiliary suspension forces (bump stop neglected  squat/lift forces neglected)
    F_SLF = p.m_s * p.g * p.b / (2 * (p.a + p.b)) - z_SLF * p.K_sf - dz_SLF * p.K_sdf + (x[6] - x[13]) * p.K_tsf / p.T_f

    F_SRF = p.m_s * p.g * p.b / (2 * (p.a + p.b)) - z_SRF * p.K_sf - dz_SRF * p.K_sdf - (x[6] - x[13]) * p.K_tsf / p.T_f

    F_SLR = p.m_s * p.g * p.a / (2 * (p.a + p.b)) - z_SLR * p.K_sr - dz_SLR * p.K_sdr + (x[6] - x[18]) * p.K_tsr / p.T_r

    F_SRR = p.m_s * p.g * p.a / (2 * (p.a + p.b)) - z_SRR * p.K_sr - dz_SRR * p.K_sdr - (x[6] - x[18]) * p.K_tsr / p.T_r

    # auxiliary variables sprung mass
    sumX = F_x_LR + F_x_RR + (F_x_LF + F_x_RF) * math.cos(x[2]) - (F_y_LF + F_y_RF) * math.sin(x[2])

    sumN = (F_y_LF + F_y_RF) * p.a * math.cos(x[2]) + (F_x_LF + F_x_RF) * p.a * math.sin(x[2]) \
           + (F_y_RF - F_y_LF) * 0.5 * p.T_f * math.sin(x[2]) + (F_x_LF - F_x_RF) * 0.5 * p.T_f * math.cos(x[2]) \
           + (F_x_LR - F_x_RR) * 0.5 * p.T_r - (F_y_LR + F_y_RR) * p.b

    sumY_s = (F_RAF + F_RAR) * math.cos(x[6]) + (F_SLF + F_SLR + F_SRF + F_SRR) * math.sin(x[6])

    sumL = 0.5 * F_SLF * p.T_f + 0.5 * F_SLR * p.T_r - 0.5 * F_SRF * p.T_f - 0.5 * F_SRR * p.T_r \
           - F_RAF / math.cos(x[6]) * (p.h_s - x[11] - p.R_w + x[16] - (p.h_raf - p.R_w) * math.cos(x[13])) \
           - F_RAR / math.cos(x[6]) * (p.h_s - x[11] - p.R_w + x[21] - (p.h_rar - p.R_w) * math.cos(x[18]))

    sumZ_s = (F_SLF + F_SLR + F_SRF + F_SRR) * math.cos(x[6]) - (F_RAF + F_RAR) * math.sin(x[6])

    sumM_s = p.a * (F_SLF + F_SRF) - p.b * (F_SLR + F_SRR) + ((F_x_LF + F_x_RF) * math.cos(x[2])
                                                              - (F_y_LF + F_y_RF) * math.sin(
                x[2]) + F_x_LR + F_x_RR) * (p.h_s - x[11])

    # auxiliary variables unsprung mass
    sumL_uf = 0.5 * F_SRF * p.T_f - 0.5 * F_SLF * p.T_f - F_RAF * (p.h_raf - p.R_w) \
              + F_z_LF * (p.R_w * math.sin(x[13]) + 0.5 * p.T_f * math.cos(x[13]) - p.K_lt * F_y_LF) \
              - F_z_RF * (-p.R_w * math.sin(x[13]) + 0.5 * p.T_f * math.cos(x[13]) + p.K_lt * F_y_RF) \
              - ((F_y_LF + F_y_RF) * math.cos(x[2]) + (F_x_LF + F_x_RF) * math.sin(x[2])) * (p.R_w - x[16])

    sumL_ur = 0.5 * F_SRR * p.T_r - 0.5 * F_SLR * p.T_r - F_RAR * (p.h_rar - p.R_w) \
              + F_z_LR * (p.R_w * math.sin(x[18]) + 0.5 * p.T_r * math.cos(x[18]) - p.K_lt * F_y_LR) \
              - F_z_RR * (-p.R_w * math.sin(x[18]) + 0.5 * p.T_r * math.cos(x[18]) + p.K_lt * F_y_RR) \
              - (F_y_LR + F_y_RR) * (p.R_w - x[21])

    sumZ_uf = F_z_LF + F_z_RF + F_RAF * math.sin(x[6]) - (F_SLF + F_SRF) * math.cos(x[6])

    sumZ_ur = F_z_LR + F_z_RR + F_RAR * math.sin(x[6]) - (F_SLR + F_SRR) * math.cos(x[6])

    sumY_uf = (F_y_LF + F_y_RF) * math.cos(x[2]) + (F_x_LF + F_x_RF) * math.sin(x[2]) \
              - F_RAF * math.cos(x[6]) - (F_SLF + F_SRF) * math.sin(x[6])

    sumY_ur = (F_y_LR + F_y_RR) \
              - F_RAR * math.cos(x[6]) - (F_SLR + F_SRR) * math.sin(x[6])

    # dynamics common with single-track model
    f = []  # init 'right hand side'
    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        # wheelbase
        # lwb = p.a + p.b

        # system dynamics
        # x_ks = [x[0],  x[1],  x[2],  x[3],  x[4]]
        # f_ks = vehicle_dynamics_ks(x_ks, u, p)
        # f.extend(f_ks)
        # f.append(u[1]*lwb*math.tan(x[2]) + x[3]/(lwb*math.cos(x[2])**2)*u[0])

        # Use kinematic model with reference point at center of mass
        # wheelbase
        lwb = p.a + p.b
        # system dynamics
        x_ks = [x[0], x[1], x[2], x[3], x[4]]
        # kinematic model
        f_ks = vehicle_dynamics_ks_cog(x_ks, u, p)
        f = [f_ks[0], f_ks[1], f_ks[2], f_ks[3], f_ks[4]]
        # derivative of slip angle and yaw rate
        d_beta = (p.b * u[0]) / (lwb * math.cos(x[2]) ** 2 * (1 + (math.tan(x[2]) ** 2 * p.b / lwb) ** 2))
        dd_psi = 1 / lwb * (u[1] * math.cos(x[6]) * math.tan(x[2]) -
                            x[3] * math.sin(x[6]) * d_beta * math.tan(x[2]) +
                            x[3] * math.cos(x[6]) * u[0] / math.cos(x[2]) ** 2)
        f.append(dd_psi)

    else:
        f.append(math.cos(beta + x[4]) * vel)
        f.append(math.sin(beta + x[4]) * vel)
        f.append(u[0])
        f.append(1 / p.m * sumX + x[5] * x[10])
        f.append(x[5])
        f.append(1 / (p.I_z - (p.I_xz_s) ** 2 / p.I_Phi_s) * (sumN + p.I_xz_s / p.I_Phi_s * sumL))

    # remaining sprung mass dynamics
    f.append(x[7])
    f.append(1 / (p.I_Phi_s - (p.I_xz_s) ** 2 / p.I_z) * (p.I_xz_s / p.I_z * sumN + sumL))
    f.append(x[9])
    f.append(1 / p.I_y_s * sumM_s)
    f.append(1 / p.m_s * sumY_s - x[5] * x[3])
    f.append(x[12])
    f.append(p.g - 1 / p.m_s * sumZ_s)

    # unsprung mass dynamics (front)
    f.append(x[14])
    f.append(1 / p.I_uf * sumL_uf)
    f.append(1 / p.m_uf * sumY_uf - x[5] * x[3])
    f.append(x[17])
    f.append(p.g - 1 / p.m_uf * sumZ_uf)

    # unsprung mass dynamics (rear)
    f.append(x[19])
    f.append(1 / p.I_ur * sumL_ur)
    f.append(1 / p.m_ur * sumY_ur - x[5] * x[3])
    f.append(x[22])
    f.append(p.g - 1 / p.m_ur * sumZ_ur)

    # convert acceleration input to brake and engine torque
    if u[1] > 0:
        T_B = 0.0
        T_E = p.m * p.R_w * u[1]
    else:
        T_B = p.m * p.R_w * u[1]
        T_E = 0.

    # wheel dynamics (p.T  new parameter for torque splitting)
    f.append(1 / p.I_y_w * (-p.R_w * F_x_LF + 0.5 * p.T_sb * T_B + 0.5 * p.T_se * T_E))
    f.append(1 / p.I_y_w * (-p.R_w * F_x_RF + 0.5 * p.T_sb * T_B + 0.5 * p.T_se * T_E))
    f.append(1 / p.I_y_w * (-p.R_w * F_x_LR + 0.5 * (1 - p.T_sb) * T_B + 0.5 * (1 - p.T_se) * T_E))
    f.append(1 / p.I_y_w * (-p.R_w * F_x_RR + 0.5 * (1 - p.T_sb) * T_B + 0.5 * (1 - p.T_se) * T_E))

    # negative wheel spin forbidden
    for iState in range(23, 27):
        if x[iState] < 0.0:
            x[iState] = 0.0
            f[iState] = 0.0

    # compliant joint equations
    f.append(dot_delta_y_f)
    f.append(dot_delta_y_r)

    return f


def CSTR_fun(x, t, alpha, beta, u, time_points):
    # x1 concentration of reactant C_a
    # x2 concentration of reactant C_b
    # x3 temperature inside the reactor T_R
    # x4 temperature of the cooling jacket T_K

    # u1 feed F
    # u2 heat flow Q_dot
    tStart = 0  # start time
    tFinal = 3  # final time
    tStep = 1e-2  # step size
    seq_len = int((tFinal - tStart) / tStep) + 1

    u1, u2 = u
    F = np.interp(t, time_points, u1)
    Q_dot = np.interp(t, time_points, u2)

    # Certain parameters
    K0_ab = 1.287e12  # K0 [h^-1]
    K0_bc = 1.287e12  # K0 [h^-1]
    K0_ad = 9.043e9  # K0 [l/mol.h]
    R_gas = 8.3144621e-3  # Universal gas constant
    E_A_ab = 9758.3 * 1.00  # * R_gas# [kj/mol]
    E_A_bc = 9758.3 * 1.00  # * R_gas# [kj/mol]
    E_A_ad = 8560.0 * 1.0  # * R_gas# [kj/mol]
    H_R_ab = 4.2  # [kj/mol A]
    H_R_bc = -11.0  # [kj/mol B] Exothermic
    H_R_ad = -41.85  # [kj/mol A] Exothermic
    Rou = 0.9342  # Density [kg/l]
    Cp = 3.01  # Specific Heat capacity [kj/Kg.K]
    Cp_k = 2.0  # Coolant heat capacity [kj/kg.k]
    A_R = 0.215  # Area of reactor wall [m^2]
    V_R = 10.01  # 0.01 # Volume of reactor [l]
    m_k = 5.0  # Coolant mass[kg]
    T_in = 130.0  # Temp of inflow [Celsius]
    K_w = 4032.0  # [kj/h.m^2.K]
    C_A0 = (5.7 + 4.5) / 2.0 * 1.0  # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]

    # States struct (optimization variables):
    C_a = x[0]
    if C_a > 2:
        C_a = 2
    elif C_a < 0.1:
        C_a = 0.1
    C_b = x[1]
    if C_b > 2:
        C_b = 2
    elif C_b < 0.1:
        C_b = 0.1
    T_R = x[2]
    if T_R > 140:
        T_R = 140
    elif T_R < 50:
        T_R = 50
    T_K = x[3]
    if T_K > 140:
        T_K = 140
    elif T_K < 50:
        T_K = 50

    # Input struct (optimization variables):

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    T_dif = T_R - T_K

    # Expressions can also be formed without beeing explicitly added to the model.
    # The main difference is that they will not be monitored and can only be used within the current file.
    K_1 = beta * K0_ab * math.exp((-E_A_ab) / ((T_R + 273.15)))
    K_2 = K0_bc * math.exp((-E_A_bc) / ((T_R + 273.15)))
    K_3 = K0_ad * math.exp((-alpha * E_A_ad) / ((T_R + 273.15)))

    # Differential equations
    dC_a = F * (C_A0 - C_a) - K_1 * C_a - K_3 * (C_a ** 2)
    dC_b = -F * C_b + K_1 * C_a - K_2 * C_b
    dT_R = ((K_1 * C_a * H_R_ab + K_2 * C_b * H_R_bc + K_3 * (C_a ** 2) * H_R_ad) / (-Rou * Cp)) + F * (T_in - T_R) + (
            ((K_w * A_R) * (-T_dif)) / (Rou * Cp * V_R))
    dT_K = (Q_dot + K_w * A_R * (T_dif)) / (m_k * Cp_k)
    return [dC_a, dC_b, dT_R, dT_K]


def quadcopter_nonlinear_fun(x, t, u, time_points):
    # Inputs: state vector (x), input vector (u)
    # Returns: time derivative of state vector (xdot)

    g = 9.81
    m = 1.
    Ix = 8.1 * 1e-3
    Iy = 8.1 * 1e-3
    Iz = 14.2 * 1e-3

    ft, tau_x, tau_y, tau_z = u
    ft = np.interp(t, time_points, ft)
    tau_x = np.interp(t, time_points, tau_x)
    tau_y = np.interp(t, time_points, tau_y)
    tau_z = np.interp(t, time_points, tau_z)

    phi = x[6]
    dphi = x[7]
    theta = x[8]
    dtheta = x[9]
    psi = x[10]
    dpsi = x[11]
    dot_x = [
        x[1],
        ft / m * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)),
        x[3],
        ft / m * (np.cos(phi) * np.sin(psi) * np.sin(theta) - np.cos(psi) * np.sin(phi)),
        x[5],
        -g + ft / m * np.cos(phi) * np.cos(theta),
        dphi,
        (Iy - Iz) / Ix * dtheta * dpsi + tau_x / Ix,
        dtheta,
        (Iz - Ix) / Iy * dphi * dpsi + tau_y / Iy,
        dpsi,
        (Ix - Iy) / Iz * dphi * dtheta + tau_z / Iz]
    return dot_x


def kinematic_single_track_model(u, t):
    x0 = [0, 0.09, 0, 0]

    sol = odeint(kST_fun, x0, t, args=(u, np.linspace(t[0], t[-1], len(u[0]))))
    return sol


def single_track_model(u, time_points):
    e0 = 1.0
    mu = 0.8
    lf = 0.5
    if vehicle == 3:
        p = parameters_vehicle3()
    elif vehicle == 2:
        p = parameters_vehicle2()
    else:
        p = parameters_vehicle1()
    p.g = 9.81  # [m/s^2]
    p.v = 70 / 3.6
    p.v_s = 0.2
    p.v_b = 0.05
    p.v_min = p.v_s / 2
    p.l = 4.976
    p.w = 1.963
    p.m_load = 150
    p.m_0 = 2108.0
    p.m = p.m_0 + p.m_load
    p.m_s_0 = 1841.9
    p.m_s = p.m_load + p.m_s_0
    p.m_uf = 121.71
    p.m_ur = 144.42
    p.a = 1.5
    p.b = 1.5
    p.l_wb = p.a + p.b
    p.I_Phi_s_0 = 370.88
    p.I_y_s_0 = 1451.5
    p.I_z_0 = 2375.54
    p.I_Phi_s = p.I_Phi_s_0 / p.m_s_0 * p.m_s
    p.I_y_s = p.I_y_s_0 / p.m_s_0 * p.m_s
    p.I_z = p.I_z_0 / p.m_0 * p.m
    p.T_f = 1.661
    p.T_r = 1.699
    p.h_cg = 0.545
    p.h_s = 0.578
    p.I_uf = 83.244
    p.I_ur = 102.147
    p.tire.p_dy1 = 1.0
    p.c_s_f0 = - p.tire.p_ky1 / p.tire.p_dy1
    p.c_s_r0 = - p.tire.p_ky1 / p.tire.p_dy1
    p.M = 0
    p.F = 0
    p.l_f = lf
    p.l_r = p.l_wb - p.l_f
    p.mu = mu
    p.tire.p_dy1 = p.mu
    p.e_0 = e0
    tStart = 0  # start time
    tFinal = 3  # final time
    tStep = 1e-2  # step size
    seq_len = int((tFinal - tStart) / tStep) + 1
    t = np.linspace(tStart, tFinal, seq_len)

    x0 = [0, e0, 0, p.v, 0, 0, 0]

    sol = odeint(ST_fun, x0, t, args=(p, u, time_points))
    return sol


def single_track_drift_model(u, time_points):
    e0 = 1.0
    mu = 0.8
    lf = 0.5
    if vehicle == 3:
        p = parameters_vehicle3()
    elif vehicle == 2:
        p = parameters_vehicle2()
    else:
        p = parameters_vehicle1()
    p.g = 9.81  # [m/s^2]
    p.v = 70 / 3.6
    p.v_s = 0.2
    p.v_b = 0.05
    p.v_min = p.v_s / 2
    p.l = 4.976
    p.w = 1.963
    p.m_load = 150
    p.m_0 = 2108.0
    p.m = p.m_0 + p.m_load
    p.m_s_0 = 1841.9
    p.m_s = p.m_load + p.m_s_0
    p.m_uf = 121.71
    p.m_ur = 144.42
    p.a = 1.5
    p.b = 1.5
    p.l_wb = p.a + p.b
    p.I_Phi_s_0 = 370.88
    p.I_y_s_0 = 1451.5
    p.I_z_0 = 2375.54
    p.I_Phi_s = p.I_Phi_s_0 / p.m_s_0 * p.m_s
    p.I_y_s = p.I_y_s_0 / p.m_s_0 * p.m_s
    p.I_z = p.I_z_0 / p.m_0 * p.m
    p.T_f = 1.661
    p.T_r = 1.699
    p.h_cg = 0.545
    p.h_s = 0.578
    p.I_uf = 83.244
    p.I_ur = 102.147
    p.tire.p_dy1 = 1.0
    p.c_s_f0 = - p.tire.p_ky1 / p.tire.p_dy1
    p.c_s_r0 = - p.tire.p_ky1 / p.tire.p_dy1
    p.M = 0
    p.F = 0
    p.l_f = lf
    p.l_r = p.l_wb - p.l_f
    p.mu = mu
    p.tire.p_dy1 = p.mu
    p.e_0 = e0
    tStart = 0  # start time
    tFinal = 3  # final time
    tStep = 1e-2  # step size
    seq_len = int((tFinal - tStart) / tStep) + 1
    t = np.linspace(tStart, tFinal, seq_len)

    x0 = [0, e0, 0, p.v, 0, 0, 0]

    sol = odeint(STD_fun, init_std(x0, p), t, args=(p, u, time_points))
    return sol


def multi_body_model(u, time_points):
    e0 = 1.0
    mu = 0.8
    lf = 0.5
    if vehicle == 3:
        p = parameters_vehicle3()
    elif vehicle == 2:
        p = parameters_vehicle2()
    else:
        p = parameters_vehicle1()
    p.g = 9.81  # [m/s^2]
    p.v = 70 / 3.6
    p.v_s = 0.2
    p.v_b = 0.05
    p.v_min = p.v_s / 2
    p.l = 4.976
    p.w = 1.963
    p.m_load = 150
    p.m_0 = 2108.0
    p.m = p.m_0 + p.m_load
    p.m_s_0 = 1841.9
    p.m_s = p.m_load + p.m_s_0
    p.m_uf = 121.71
    p.m_ur = 144.42
    p.a = 1.5
    p.b = 1.5
    p.l_wb = p.a + p.b
    p.I_Phi_s_0 = 370.88
    p.I_y_s_0 = 1451.5
    p.I_z_0 = 2375.54
    p.I_Phi_s = p.I_Phi_s_0 / p.m_s_0 * p.m_s
    p.I_y_s = p.I_y_s_0 / p.m_s_0 * p.m_s
    p.I_z = p.I_z_0 / p.m_0 * p.m
    p.T_f = 1.661
    p.T_r = 1.699
    p.h_cg = 0.545
    p.h_s = 0.578
    p.I_uf = 83.244
    p.I_ur = 102.147
    p.tire.p_dy1 = 1.0
    p.c_s_f0 = - p.tire.p_ky1 / p.tire.p_dy1
    p.c_s_r0 = - p.tire.p_ky1 / p.tire.p_dy1
    p.M = 0
    p.F = 0
    p.l_f = lf
    p.l_r = p.l_wb - p.l_f
    p.mu = mu
    p.tire.p_dy1 = p.mu
    p.e_0 = e0
    tStart = 0  # start time
    tFinal = 3  # final time
    tStep = 1e-2  # step size
    seq_len = int((tFinal - tStart) / tStep) + 1
    t = np.linspace(tStart, tFinal, seq_len)

    x0 = [0, e0, 0, p.v, 0, 0, 0]

    sol = odeint(MB_fun, init_mb(x0, p), t, args=(p, u, time_points))
    return sol


def CSTR_model(u, time_points):
    tStart = 0  # start time
    tFinal = 3  # final time
    tStep = 1e-2  # step size
    seq_len = int((tFinal - tStart) / tStep) + 1
    t = np.linspace(tStart, tFinal, seq_len)

    C_a_0 = 0.8  # This is the initial concentration inside the tank [mol/l]
    C_b_0 = 0.5  # This is the controlled variable [mol/l]
    T_R_0 = 134.14  # [C]
    T_K_0 = 130.0  # [C]
    x0 = [C_a_0, C_b_0, T_R_0, T_K_0]

    alpha = 1.0
    beta = 1.0

    sol = odeint(CSTR_fun, x0, t, args=(alpha, beta, u, time_points))
    return sol


def quadcopter_model(u, time_points):
    tStart = 0  # start time
    tFinal = 3  # final time
    tStep = 0.01  # step size
    seq_len = int((tFinal - tStart) / tStep) + 1
    t = np.linspace(tStart, tFinal, seq_len)
    x0 = list(np.zeros(12))
    sol = odeint(quadcopter_nonlinear_fun, x0, t, args=(u, time_points))
    return sol


def train_data_generator(batch_size, func=None):
    a1 = np.random.uniform(0.05, 0.5, (batch_size // 10,))
    a2 = np.random.uniform(0.05, 1.0, (batch_size // 10,))
    w1 = np.random.uniform(0.5, 2.5, (batch_size // 10,))
    f0 = np.random.uniform(0.05, 0.5, (batch_size // 10,))
    f1 = np.random.uniform(2.5, 4.5, (batch_size // 10,))

    f2 = np.random.uniform(0.5, 5, (2 * (batch_size // 5),))
    h2 = np.random.uniform(0.05, 0.5, (2 * (batch_size // 5),))
    f3 = np.random.uniform(0.5, 5, (2 * (batch_size // 5),))
    h3 = np.random.uniform(0.05, 1.0, (2 * (batch_size // 5),))

    k1 = np.random.uniform(0.5, 1.0, (batch_size // 10,))
    j1 = np.random.uniform(0.05, 0.5, (batch_size // 10,))
    k2 = np.random.uniform(1, 5, (batch_size // 10,))
    j2 = np.random.uniform(0.05, 1.0, (batch_size // 10,))

    b1 = np.random.uniform(0.05, 0.5, (2 * (batch_size // 5),))
    b2 = np.random.uniform(0.05, 1.0, (2 * (batch_size // 5),))
    v1 = np.random.uniform(0.5, 2.5, (2 * (batch_size // 5),))
    v2 = np.random.uniform(2.5, 4.5, (2 * (batch_size // 5),))
    tStart = 0  # start time
    tFinal = 3  # final time
    tStep = 1e-2  # step size
    seq_len = int((tFinal - tStart) / tStep) + 1
    t = np.linspace(tStart, tFinal, seq_len)
    X_batch = []
    u_batch = []
    num = 0
    for i in range(batch_size // 20):
        num += 1
        print(num)
        u1 = a1[i] * np.sin(2 * np.pi * w1[i] * t)
        u2 = a2[i] * scisi.chirp(t, f0[i], 3, f1[i])
        u_batch += [np.stack([u1, u2], axis=-1)]
        x = func([u1, u2], t)
        X_batch += [x]
    for i in range(batch_size // 20, batch_size // 10):
        num += 1
        print(num)
        u1 = a1[i] * np.cos(2 * np.pi * w1[i] * t)
        u2 = a2[i] * scisi.chirp(t, f1[i], 3, f0[i])
        u_batch += [np.stack([u1, u2], axis=-1)]
        x = func([u1, u2], t)
        X_batch += [x]
    for i in range(batch_size // 10):
        num += 1
        print(num)
        u1 = h2[i] * np.abs(scisi.sawtooth(2 * f2[i] * t)) - h2[i] / 2
        u2 = h3[i] * scisi.square(2 * f3[i] * t)
        u_batch += [np.stack([u1, u2], axis=-1)]
        x = func([u1, u2], t)
        X_batch += [x]
    for i in range(batch_size // 10, batch_size // 5):
        num += 1
        print(num)
        u1 = h2[i] * scisi.square(2 * f2[i] * t)
        u2 = h3[i] * np.abs(scisi.sawtooth(2 * f3[i] * t)) - h3[i] / 2
        u_batch += [np.stack([u1, u2], axis=-1)]
        x = func([u1, u2], t)
        X_batch += [x]
    for i in range(batch_size // 5, 3 * (batch_size // 10)):
        num += 1
        print(num)
        u1 = h2[i] * scisi.sawtooth(2 * f2[i] * t)
        u2 = np.concatenate([h3[i] / (j + 1) * np.ones(seq_len // 5) for j in range(4)] + [
            h3[i] / 5 * np.ones(seq_len - 4 * (seq_len // 5))])
        u_batch += [np.stack([u1, u2], axis=-1)]
        x = func([u1, u2], t)
        X_batch += [x]
    for i in range(3 * (batch_size // 10), 2 * (batch_size // 5)):
        num += 1
        print(num)
        u1 = np.concatenate([h2[i] / (j + 1) * np.ones(seq_len // 5) for j in range(4)] + [
            h2[i] / 5 * np.ones(seq_len - 4 * (seq_len // 5))])
        u2 = h3[i] * scisi.sawtooth(2 * f3[i] * t)
        u_batch += [np.stack([u1, u2], axis=-1)]
        x = func([u1, u2], t)
        X_batch += [x]
    for i in range(batch_size // 20):
        num += 1
        print(num)
        u1 = j1[i] * np.exp(- k1[i] * t)
        u2 = j2[i] * (1 - np.exp(- k2[i] * t))
        u_batch += [np.stack([u1, u2], axis=-1)]
        x = func([u1, u2], t)
        X_batch += [x]
    for i in range(batch_size // 20, batch_size // 10):
        num += 1
        print(num)
        u1 = j1[i] * (1 - np.exp(- k1[i] * t))
        u2 = j2[i] * np.exp(- k2[i] * t)
        u_batch += [np.stack([u1, u2], axis=-1)]
        x = func([u1, u2], t)
        X_batch += [x]
    for i in range(batch_size // 5):
        num += 1
        print(num)
        q1, q2, _ = scisi.gausspulse(t, fc=v1[i], retquad=True, retenv=True)
        u1 = b1[i] * q1
        u2 = b2[i] * q2
        u_batch += [np.stack([u1, u2], axis=-1)]
        x = func([u1, u2], t)
        X_batch += [x]
    for i in range(batch_size // 5, 2 * (batch_size // 5)):
        num += 1
        print(num)
        q1, q2, _ = scisi.gausspulse(t, fc=v2[i], retquad=True, retenv=True)
        u1 = b1[i] * q2
        u2 = b2[i] * q1
        u_batch += [np.stack([u1, u2], axis=-1)]
        x = func([u1, u2], t)
        X_batch += [x]
    X_batch = np.stack(X_batch, 0)
    u_batch = np.stack(u_batch, 0)
    return u_batch, X_batch


def cstr_data_generator(batch_size, func=CSTR_model):
    tStart = 0  # start time
    tFinal = 3  # final time
    tStep = 1e-2  # step size
    seq_len = int((tFinal - tStart) / tStep) + 1
    u1_min = 15
    u1_max = 40
    u2_min = -8000
    u2_max = -1600
    a1 = np.random.uniform(0.05, 0.1, (batch_size,))
    a2 = np.random.uniform(0.1, 10, (batch_size,))
    b1 = np.random.uniform(u1_min, u1_max, (batch_size,))
    b2 = np.random.uniform(u2_min, u2_max, (batch_size,))

    X_batch = []
    u_batch = []
    for i in range(batch_size // 4):
        print(i)
        u1 = []
        u2 = []
        for j in range(9):
            _u1 = max(b1[i] - a1[i] * j, u1_min)
            u1.append(_u1 * np.ones(seq_len // 10))
            _u2 = min(b2[i] + a2[i] * j, u2_max)
            u2.append(_u2 * np.ones(seq_len // 10))
        u1.append(_u1 * np.ones(seq_len - 9 * (seq_len // 10)))
        u2.append(_u2 * np.ones(seq_len - 9 * (seq_len // 10)))
        u_batch += [np.stack([np.concatenate(u1), np.concatenate(u2)], axis=-1)]
        x = func([a1[i], a2[i], b1[i], b2[i], u1_min, u1_max, u2_min, u2_max], 0)
        X_batch += [x]
    for i in range(batch_size // 4, batch_size // 2):
        print(i)
        u1 = []
        u2 = []
        for j in range(9):
            _u1 = min(b1[i] + a1[i] * j, u1_max)
            u1.append(_u1 * np.ones(seq_len // 10))
            _u2 = max(b2[i] - a2[i] * j, u2_min)
            u2.append(_u2 * np.ones(seq_len // 10))
        u1.append(_u1 * np.ones(seq_len - 9 * (seq_len // 10)))
        u2.append(_u2 * np.ones(seq_len - 9 * (seq_len // 10)))
        u_batch += [np.stack([np.concatenate(u1), np.concatenate(u2)], axis=-1)]
        x = func([a1[i], a2[i], b1[i], b2[i], u1_min, u1_max, u2_min, u2_max], 1)
        X_batch += [x]
    for i in range(batch_size // 2, 3 * (batch_size // 4)):
        print(i)
        u1 = []
        u2 = []
        for j in range(9):
            _u1 = min(b1[i] + a1[i] * j, u1_max)
            u1.append(_u1 * np.ones(seq_len // 10))
            _u2 = min(b2[i] + a2[i] * j, u2_max)
            u2.append(_u2 * np.ones(seq_len // 10))
        u1.append(_u1 * np.ones(seq_len - 9 * (seq_len // 10)))
        u2.append(_u2 * np.ones(seq_len - 9 * (seq_len // 10)))
        u_batch += [np.stack([np.concatenate(u1), np.concatenate(u2)], axis=-1)]
        x = func([a1[i], a2[i], b1[i], b2[i], u1_min, u1_max, u2_min, u2_max], 2)
        X_batch += [x]
    for i in range(3 * (batch_size // 4), batch_size):
        print(i)
        u1 = []
        u2 = []
        for j in range(9):
            _u1 = max(b1[i] - a1[i] * j, u1_min)
            u1.append(_u1 * np.ones(seq_len // 10))
            _u2 = max(b2[i] - a2[i] * j, u2_min)
            u2.append(_u2 * np.ones(seq_len // 10))
        u1.append(_u1 * np.ones(seq_len - 9 * (seq_len // 10)))
        u2.append(_u2 * np.ones(seq_len - 9 * (seq_len // 10)))
        u_batch += [np.stack([np.concatenate(u1), np.concatenate(u2)], axis=-1)]
        x = func([a1[i], a2[i], b1[i], b2[i], u1_min, u1_max, u2_min, u2_max], 3)
        X_batch += [x]
    X_batch = np.stack(X_batch, 0)
    u_batch = np.stack(u_batch, 0)
    return u_batch, X_batch


def quadcopter_data_generator(require_num, func=quadcopter_model):
    l = 0.5
    wmin = 0
    wmax = 5
    tStart = 0  # start time
    tFinal = 3  # final time
    tStep = 0.01  # step size
    seq_len = int((tFinal - tStart) / tStep) + 1

    t = np.linspace(tStart, tFinal, seq_len)
    X_batch = []
    u_batch = []
    success_num = 0
    while success_num < require_num:
        a1 = np.random.uniform(wmin, wmax, (4))
        f1 = np.random.uniform(0.5, 5, (4))
        if success_num % 4 == 0:
            w1 = a1[0] * np.ones(seq_len)
            w2 = a1[1] * np.ones(seq_len)
            w3 = a1[2] * np.ones(seq_len)
            w4 = a1[3] * np.ones(seq_len)
        elif success_num % 4 == 1:
            w1 = a1[0] * scisi.square(2 * f1[0] * t)
            w2 = a1[1] * scisi.square(2 * f1[1] * t)
            w3 = a1[2] * scisi.square(2 * f1[2] * t)
            w4 = a1[3] * scisi.square(2 * f1[3] * t)
        elif success_num % 4 == 2:
            w1 = np.concatenate([a1[0] / (5 - j) * np.ones(seq_len // 5) for j in range(4)] + [
                a1[0] * np.ones(seq_len - 4 * (seq_len // 5))])
            w2 = np.concatenate([a1[1] / (5 - j) * np.ones(seq_len // 5) for j in range(4)] + [
                a1[1] * np.ones(seq_len - 4 * (seq_len // 5))])
            w3 = np.concatenate([a1[2] / (5 - j) * np.ones(seq_len // 5) for j in range(4)] + [
                a1[2] * np.ones(seq_len - 4 * (seq_len // 5))])
            w4 = np.concatenate([a1[3] / (5 - j) * np.ones(seq_len // 5) for j in range(4)] + [
                a1[3] * np.ones(seq_len - 4 * (seq_len // 5))])
        elif success_num % 4 == 3:
            w1 = np.concatenate([a1[0] / (j + 1) * np.ones(seq_len // 5) for j in range(4)] + [
                a1[0] / 5 * np.ones(seq_len - 4 * (seq_len // 5))])
            w2 = np.concatenate([a1[1] / (j + 1) * np.ones(seq_len // 5) for j in range(4)] + [
                a1[1] / 5 * np.ones(seq_len - 4 * (seq_len // 5))])
            w3 = np.concatenate([a1[2] / (j + 1) * np.ones(seq_len // 5) for j in range(4)] + [
                a1[2] / 5 * np.ones(seq_len - 4 * (seq_len // 5))])
            w4 = np.concatenate([a1[3] / (j + 1) * np.ones(seq_len // 5) for j in range(4)] + [
                a1[3] / 5 * np.ones(seq_len - 4 * (seq_len // 5))])

        u1 = (w1 ** 2 + w2 ** 2 + w3 ** 2 + w4 ** 2)
        u2 = l * (w4 ** 2 - w2 ** 2)
        u3 = l * (w3 ** 2 - w1 ** 2)
        u4 = l * (w1 ** 2 + w3 ** 2 - w2 ** 2 - w4 ** 2)
        x = func([u1, u2, u3, u4], t)
        X_batch.append(x)
        u_batch.append(np.stack([w1, w2, w3, w4], axis=-1))
        success_num += 1
        print(success_num)

    X_batch = np.stack(X_batch, 0)
    u_batch = np.stack(u_batch, 0)
    return u_batch, X_batch


if __name__ == '__main__':

    data_path = 'data'
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    # num = 500
    # u, X = train_data_generator(num, func=single_track_model)
    # np.save(os.path.join(data_path, fr'single_track_{num}_u_val.npy'), u)
    # np.save(os.path.join(data_path, fr'single_track_{num}_x_val.npy'), X)
    # num = 500
    # u, X = train_data_generator(num, func=single_track_model)
    # np.save(os.path.join(data_path, fr'single_track_{num}_u_test.npy'), u)
    # np.save(os.path.join(data_path, fr'single_track_{num}_x_test.npy'), X)
    # num = 4000
    # u, X = train_data_generator(num, func=single_track_model)
    # np.save(os.path.join(data_path, fr'single_track_{num}_u_train.npy'), u)
    # np.save(os.path.join(data_path, fr'single_track_{num}_x_train.npy'), X)

    # num = 500
    # u, X = train_data_generator(num, func=single_track_drift_model)
    # np.save(os.path.join(data_path, fr'single_track_drift_{num}_u_val.npy'), u)
    # np.save(os.path.join(data_path, fr'single_track_drift_{num}_x_val.npy'), X)
    # num = 500
    # u, X = train_data_generator(num, func=single_track_drift_model)
    # np.save(os.path.join(data_path, fr'single_track_drift_{num}_u_test.npy'), u)
    # np.save(os.path.join(data_path, fr'single_track_drift_{num}_x_test.npy'), X)
    # num = 4000
    # u, X = train_data_generator(num, func=single_track_drift_model)
    # np.save(os.path.join(data_path, fr'single_track_drift_{num}_u_train.npy'), u)
    # np.save(os.path.join(data_path, fr'single_track_drift_{num}_x_train.npy'), X)

    num = 500
    u, X = train_data_generator(num, func=multi_body_model)
    np.save(os.path.join(data_path, fr'multi_body_{num}_u_val.npy'), u)
    np.save(os.path.join(data_path, fr'multi_body_{num}_x_val.npy'), X)
    num = 500
    u, X = train_data_generator(num, func=multi_body_model)
    np.save(os.path.join(data_path, fr'multi_body_{num}_u_test.npy'), u)
    np.save(os.path.join(data_path, fr'multi_body_{num}_x_test.npy'), X)
    num = 4000
    u, X = train_data_generator(num, func=multi_body_model)
    np.save(os.path.join(data_path, fr'multi_body_{num}_u_train.npy'), u)
    np.save(os.path.join(data_path, fr'multi_body_{num}_x_train.npy'), X)

