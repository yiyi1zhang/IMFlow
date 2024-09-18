import torch
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
sys.path.append('../../../..')
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3

def model_parameters():
    e0 = 1.0
    mu = 0.8
    lf = 0.5
    p = parameters_vehicle3()
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

    return p

def formula_longitudinal(kappa, gamma, F_z, p):
    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    # coordinate system transformation
    kappa = -kappa

    S_hx = p.p_hx1
    S_vx = F_z * p.p_vx1

    kappa_x = kappa + S_hx
    mu_x = p.p_dx1 * (1 - p.p_dx3 * gamma ** 2)

    C_x = p.p_cx1
    D_x = mu_x * F_z
    E_x = p.p_ex1
    K_x = F_z * p.p_kx1
    B_x = K_x / (C_x * D_x)

    # magic tire formula
    return D_x * torch.sin(C_x * torch.atan(B_x * kappa_x - E_x * (B_x * kappa_x - torch.atan(B_x * kappa_x))) + S_vx)


# lateral tire forces
def formula_lateral(alpha, gamma, F_z, p):
    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    # coordinate system transformation
    # alpha = -alpha

    S_hy = torch.sign(gamma) * (p.p_hy1 + p.p_hy3 * torch.abs(gamma))
    S_vy = torch.sign(gamma) * F_z * (p.p_vy1 + p.p_vy3 * torch.abs(gamma))

    alpha_y = alpha + S_hy
    mu_y = p.p_dy1 * (1 - p.p_dy3 * gamma ** 2)

    C_y = p.p_cy1
    D_y = mu_y * F_z
    E_y = p.p_ey1
    K_y = F_z * p.p_ky1  # simplify K_y0 to p.p_ky1*F_z
    B_y = K_y / (C_y * D_y)

    # magic tire formula
    F_y = D_y * torch.sin(C_y * torch.atan(B_y * alpha_y - E_y * (B_y * alpha_y - torch.atan(B_y * alpha_y)))) + S_vy
    res = [F_y, mu_y]
    return res


# longitudinal tire forces for combined slip
def formula_longitudinal_comb(kappa, alpha, F0_x, p):
    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    S_hxalpha = p.r_hx1

    alpha_s = alpha + S_hxalpha

    B_xalpha = p.r_bx1 * torch.cos(torch.atan(p.r_bx2 * kappa))
    C_xalpha = p.r_cx1
    E_xalpha = p.r_ex1
    D_xalpha = F0_x / (torch.cos(C_xalpha * torch.atan(
        B_xalpha * S_hxalpha - E_xalpha * (B_xalpha * S_hxalpha - torch.atan(B_xalpha * S_hxalpha)))))

    # magic tire formula
    return D_xalpha * torch.cos(
        C_xalpha * torch.atan(B_xalpha * alpha_s - E_xalpha * (B_xalpha * alpha_s - torch.atan(B_xalpha * alpha_s))))


# lateral tire forces for combined slip
def formula_lateral_comb(kappa, alpha, gamma, mu_y, F_z, F0_y, p):
    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    S_hykappa = p.r_hy1

    kappa_s = kappa + S_hykappa

    B_ykappa = p.r_by1 * torch.cos(torch.atan(p.r_by2 * (alpha - p.r_by3)))
    C_ykappa = p.r_cy1
    E_ykappa = p.r_ey1
    D_ykappa = F0_y / (torch.cos(C_ykappa * torch.atan(
        B_ykappa * S_hykappa - E_ykappa * (B_ykappa * S_hykappa - torch.atan(B_ykappa * S_hykappa)))))

    D_vykappa = mu_y * F_z * (p.r_vy1 + p.r_vy3 * gamma) * torch.cos(torch.atan(p.r_vy4 * alpha))
    S_vykappa = D_vykappa * torch.sin(p.r_vy5 * torch.atan(p.r_vy6 * kappa))

    # magic tire formula
    return D_ykappa * torch.cos(C_ykappa * torch.atan(
        B_ykappa * kappa_s - E_ykappa * (B_ykappa * kappa_s - torch.atan(B_ykappa * kappa_s)))) + S_vykappa


p = model_parameters()

class SingleTrackDynamics:

    def __init__(self, batch_size=1):
        self.batch_size = batch_size

    def __call__(self, state, action, dt):
        """
        Compute new state from state and action
        """

        dx = state[..., 2] * torch.cos(state[..., 3])
        dy = state[..., 2] * torch.sin(state[..., 3])
        dv = action[..., 1]
        dphi = state[..., 2] * torch.tan(action[..., 0]) / p.l_wb

        new_x = state[..., 0] + dx * dt
        new_y = state[..., 1] + dy * dt
        new_v = state[..., 2] + dv * dt
        new_phi = state[..., 3] + dphi * dt

        next_state = torch.stack(
            [new_x, new_y, new_v, new_phi], dim=-1
        )
        return next_state

    def loss(self, state, action, ref, dims, factors, eval_act=True, mean=True):
        Rd = [0.1, 0.1] # input difference cost matrix
        Q = [10, 1000, 0, 0]
        loss = 0
        for i in range(len(Q)):
            loss += Q[i] * (state[..., dims[i]] - ref[..., i])**2
        if eval_act:
            for i in range(len(Rd)):
                loss += Rd[i] * (action[..., i]**2)

        return torch.mean(loss) if mean else loss

class MultiBodyDynamics:

    def __init__(self, batch_size=1):
        self.batch_size = batch_size

    def __call__(self, state, action, dt):
        """
        Compute new state from state and action
        """
        inds = torch.where(torch.abs(state[:, 3]) < 0.1)[0]
        beta = torch.atan(state[..., 10] / state[..., 3])
        beta[inds] = 0
        vel = torch.sqrt(state[..., 3] ** 2 + state[..., 10] ** 2)
        vel[inds] = state[inds, 3]

        # vertical tire forces
        F_z_LF = (state[..., 16] + p.R_w * (torch.cos(state[..., 13]) - 1) - 0.5 * p.T_f * torch.sin(state[..., 13])) * p.K_zt
        F_z_RF = (state[..., 16] + p.R_w * (torch.cos(state[..., 13]) - 1) + 0.5 * p.T_f * torch.sin(state[..., 13])) * p.K_zt
        F_z_LR = (state[..., 21] + p.R_w * (torch.cos(state[..., 18]) - 1) - 0.5 * p.T_r * torch.sin(state[..., 18])) * p.K_zt
        F_z_RR = (state[..., 21] + p.R_w * (torch.cos(state[..., 18]) - 1) + 0.5 * p.T_r * torch.sin(state[..., 18])) * p.K_zt

        # obtain individual tire speeds
        u_w_lf = (state[..., 3] + 0.5 * p.T_f * state[..., 5]) * torch.cos(state[..., 2]) + (state[..., 10] + p.a * state[..., 5]) * torch.sin(state[..., 2])
        u_w_rf = (state[..., 3] - 0.5 * p.T_f * state[..., 5]) * torch.cos(state[..., 2]) + (state[..., 10] + p.a * state[..., 5]) * torch.sin(state[..., 2])
        u_w_lr = state[..., 3] + 0.5 * p.T_r * state[..., 5]
        u_w_rr = state[..., 3] - 0.5 * p.T_r * state[..., 5]
        u_w_lf[torch.where(u_w_lf < 0)[0]] = 0
        u_w_rf[torch.where(u_w_rf < 0)[0]] = 0
        u_w_lr[torch.where(u_w_lr < 0)[0]] = 0
        u_w_rr[torch.where(u_w_rr < 0)[0]] = 0

        # compute longitudinal slip
        # switch to kinematic model for small velocities
        s_lf = 1 - p.R_w * state[..., 23] / u_w_lf
        s_rf = 1 - p.R_w * state[..., 24] / u_w_rf
        s_lr = 1 - p.R_w * state[..., 25] / u_w_lr
        s_rr = 1 - p.R_w * state[..., 26] / u_w_rr
        s_lf[inds] = 0
        s_rf[inds] = 0
        s_lr[inds] = 0
        s_rr[inds] = 0

            # lateral slip angles
        # switch to kinematic model for small velocities
        alpha_LF = torch.atan((state[..., 10] + p.a * state[..., 5] - state[..., 14] * (p.R_w - state[..., 16])) / (state[..., 3] + 0.5 * p.T_f * state[..., 5])) - state[..., 2]
        alpha_RF = torch.atan((state[..., 10] + p.a * state[..., 5] - state[..., 14] * (p.R_w - state[..., 16])) / (state[..., 3] - 0.5 * p.T_f * state[..., 5])) - state[..., 2]
        alpha_LR = torch.atan((state[..., 10] - p.b * state[..., 5] - state[..., 19] * (p.R_w - state[..., 21])) / (state[..., 3] + 0.5 * p.T_r * state[..., 5]))
        alpha_RR = torch.atan((state[..., 10] - p.b * state[..., 5] - state[..., 19] * (p.R_w - state[..., 21])) / (state[..., 3] - 0.5 * p.T_r * state[..., 5]))
        alpha_LF[inds] = 0
        alpha_RF[inds] = 0
        alpha_LR[inds] = 0
        alpha_RR[inds] = 0

            # auxiliary suspension movement
        z_SLF = (p.h_s - p.R_w + state[..., 16] - state[..., 11]) / torch.cos(state[..., 6]) - p.h_s + p.R_w + p.a * state[..., 8] + 0.5 * (
                    state[..., 6] - state[..., 13]) * p.T_f
        z_SRF = (p.h_s - p.R_w + state[..., 16] - state[..., 11]) / torch.cos(state[..., 6]) - p.h_s + p.R_w + p.a * state[..., 8] - 0.5 * (
                    state[..., 6] - state[..., 13]) * p.T_f
        z_SLR = (p.h_s - p.R_w + state[..., 21] - state[..., 11]) / torch.cos(state[..., 6]) - p.h_s + p.R_w - p.b * state[..., 8] + 0.5 * (
                    state[..., 6] - state[..., 18]) * p.T_r
        z_SRR = (p.h_s - p.R_w + state[..., 21] - state[..., 11]) / torch.cos(state[..., 6]) - p.h_s + p.R_w - p.b * state[..., 8] - 0.5 * (
                    state[..., 6] - state[..., 18]) * p.T_r

        dz_SLF = state[..., 17] - state[..., 12] + p.a * state[..., 9] + 0.5 * (state[..., 7] - state[..., 14]) * p.T_f
        dz_SRF = state[..., 17] - state[..., 12] + p.a * state[..., 9] - 0.5 * (state[..., 7] - state[..., 14]) * p.T_f
        dz_SLR = state[..., 22] - state[..., 12] - p.b * state[..., 9] + 0.5 * (state[..., 7] - state[..., 19]) * p.T_r
        dz_SRR = state[..., 22] - state[..., 12] - p.b * state[..., 9] - 0.5 * (state[..., 7] - state[..., 19]) * p.T_r

        # camber angles
        gamma_LF = state[..., 6] + p.D_f * z_SLF + p.E_f * (z_SLF) ** 2
        gamma_RF = state[..., 6] - p.D_f * z_SRF - p.E_f * (z_SRF) ** 2
        gamma_LR = state[..., 6] + p.D_r * z_SLR + p.E_r * (z_SLR) ** 2
        gamma_RR = state[..., 6] - p.D_r * z_SRR - p.E_r * (z_SRR) ** 2

        # compute longitudinal tire forces using the magic formula for pure slip
        F0_x_LF = formula_longitudinal(s_lf, gamma_LF, F_z_LF, p.tire)
        F0_x_RF = formula_longitudinal(s_rf, gamma_RF, F_z_RF, p.tire)
        F0_x_LR = formula_longitudinal(s_lr, gamma_LR, F_z_LR, p.tire)
        F0_x_RR = formula_longitudinal(s_rr, gamma_RR, F_z_RR, p.tire)

        # compute lateral tire forces using the magic formula for pure slip
        res = formula_lateral(alpha_LF, gamma_LF, F_z_LF, p.tire)
        F0_y_LF = res[0]
        mu_y_LF = res[1]
        res = formula_lateral(alpha_RF, gamma_RF, F_z_RF, p.tire)
        F0_y_RF = res[0]
        mu_y_RF = res[1]
        res = formula_lateral(alpha_LR, gamma_LR, F_z_LR, p.tire)
        F0_y_LR = res[0]
        mu_y_LR = res[1]
        res = formula_lateral(alpha_RR, gamma_RR, F_z_RR, p.tire)
        F0_y_RR = res[0]
        mu_y_RR = res[1]

        # compute longitudinal tire forces using the magic formula for combined slip
        F_x_LF = formula_longitudinal_comb(s_lf, alpha_LF, F0_x_LF, p.tire)
        F_x_RF = formula_longitudinal_comb(s_rf, alpha_RF, F0_x_RF, p.tire)
        F_x_LR = formula_longitudinal_comb(s_lr, alpha_LR, F0_x_LR, p.tire)
        F_x_RR = formula_longitudinal_comb(s_rr, alpha_RR, F0_x_RR, p.tire)

        # compute lateral tire forces using the magic formula for combined slip
        F_y_LF = formula_lateral_comb(s_lf, alpha_LF, gamma_LF, mu_y_LF, F_z_LF, F0_y_LF, p.tire)
        F_y_RF = formula_lateral_comb(s_rf, alpha_RF, gamma_RF, mu_y_RF, F_z_RF, F0_y_RF, p.tire)
        F_y_LR = formula_lateral_comb(s_lr, alpha_LR, gamma_LR, mu_y_LR, F_z_LR, F0_y_LR, p.tire)
        F_y_RR = formula_lateral_comb(s_rr, alpha_RR, gamma_RR, mu_y_RR, F_z_RR, F0_y_RR, p.tire)

        # auxiliary movements for compliant joint equations
        delta_z_f = p.h_s - p.R_w + state[..., 16] - state[..., 11]
        delta_z_r = p.h_s - p.R_w + state[..., 21] - state[..., 11]

        delta_phi_f = state[..., 6] - state[..., 13]
        delta_phi_r = state[..., 6] - state[..., 18]

        dot_delta_phi_f = state[..., 7] - state[..., 14]
        dot_delta_phi_r = state[..., 7] - state[..., 19]

        dot_delta_z_f = state[..., 17] - state[..., 12]
        dot_delta_z_r = state[..., 22] - state[..., 12]

        dot_delta_y_f = state[..., 10] + p.a * state[..., 5] - state[..., 15]
        dot_delta_y_r = state[..., 10] - p.b * state[..., 5] - state[..., 20]

        delta_f = delta_z_f * torch.sin(state[..., 6]) - state[..., 27] * torch.cos(state[..., 6]) - (p.h_raf - p.R_w) * torch.sin(delta_phi_f)
        delta_r = delta_z_r * torch.sin(state[..., 6]) - state[..., 28] * torch.cos(state[..., 6]) - (p.h_rar - p.R_w) * torch.sin(delta_phi_r)

        dot_delta_f = (delta_z_f * torch.cos(state[..., 6]) + state[..., 27] * torch.sin(state[..., 6])) * state[..., 7] + dot_delta_z_f * torch.sin(
            state[..., 6]) - dot_delta_y_f * torch.cos(state[..., 6]) - (p.h_raf - p.R_w) * torch.cos(delta_phi_f) * dot_delta_phi_f
        dot_delta_r = (delta_z_r * torch.cos(state[..., 6]) + state[..., 28] * torch.sin(state[..., 6])) * state[..., 7] + dot_delta_z_r * torch.sin(
            state[..., 6]) - dot_delta_y_r * torch.cos(state[..., 6]) - (p.h_rar - p.R_w) * torch.cos(delta_phi_r) * dot_delta_phi_r

        # compliant joint forces
        F_RAF = delta_f * p.K_ras + dot_delta_f * p.K_rad
        F_RAR = delta_r * p.K_ras + dot_delta_r * p.K_rad

        # auxiliary suspension forces (bump stop neglected  squat/lift forces neglected)
        F_SLF = p.m_s * p.g * p.b / (2 * (p.a + p.b)) - z_SLF * p.K_sf - dz_SLF * p.K_sdf + (
                    state[..., 6] - state[..., 13]) * p.K_tsf / p.T_f

        F_SRF = p.m_s * p.g * p.b / (2 * (p.a + p.b)) - z_SRF * p.K_sf - dz_SRF * p.K_sdf - (
                    state[..., 6] - state[..., 13]) * p.K_tsf / p.T_f

        F_SLR = p.m_s * p.g * p.a / (2 * (p.a + p.b)) - z_SLR * p.K_sr - dz_SLR * p.K_sdr + (
                    state[..., 6] - state[..., 18]) * p.K_tsr / p.T_r

        F_SRR = p.m_s * p.g * p.a / (2 * (p.a + p.b)) - z_SRR * p.K_sr - dz_SRR * p.K_sdr - (
                    state[..., 6] - state[..., 18]) * p.K_tsr / p.T_r

        # auxiliary variables sprung mass
        sumX = F_x_LR + F_x_RR + (F_x_LF + F_x_RF) * torch.cos(state[..., 2]) - (F_y_LF + F_y_RF) * torch.sin(state[..., 2])

        sumN = (F_y_LF + F_y_RF) * p.a * torch.cos(state[..., 2]) + (F_x_LF + F_x_RF) * p.a * torch.sin(state[..., 2]) \
               + (F_y_RF - F_y_LF) * 0.5 * p.T_f * torch.sin(state[..., 2]) + (F_x_LF - F_x_RF) * 0.5 * p.T_f * torch.cos(state[..., 2]) \
               + (F_x_LR - F_x_RR) * 0.5 * p.T_r - (F_y_LR + F_y_RR) * p.b

        sumY_s = (F_RAF + F_RAR) * torch.cos(state[..., 6]) + (F_SLF + F_SLR + F_SRF + F_SRR) * torch.sin(state[..., 6])

        sumL = 0.5 * F_SLF * p.T_f + 0.5 * F_SLR * p.T_r - 0.5 * F_SRF * p.T_f - 0.5 * F_SRR * p.T_r \
               - F_RAF / torch.cos(state[..., 6]) * (p.h_s - state[..., 11] - p.R_w + state[..., 16] - (p.h_raf - p.R_w) * torch.cos(state[..., 13])) \
               - F_RAR / torch.cos(state[..., 6]) * (p.h_s - state[..., 11] - p.R_w + state[..., 21] - (p.h_rar - p.R_w) * torch.cos(state[..., 18]))

        sumZ_s = (F_SLF + F_SLR + F_SRF + F_SRR) * torch.cos(state[..., 6]) - (F_RAF + F_RAR) * torch.sin(state[..., 6])

        sumM_s = p.a * (F_SLF + F_SRF) - p.b * (F_SLR + F_SRR) + ((F_x_LF + F_x_RF) * torch.cos(state[..., 2])
                                                                  - (F_y_LF + F_y_RF) * torch.sin(
                    state[..., 2]) + F_x_LR + F_x_RR) * (p.h_s - state[..., 11])

        # auxiliary variables unsprung mass
        sumL_uf = 0.5 * F_SRF * p.T_f - 0.5 * F_SLF * p.T_f - F_RAF * (p.h_raf - p.R_w) \
                  + F_z_LF * (p.R_w * torch.sin(state[..., 13]) + 0.5 * p.T_f * torch.cos(state[..., 13]) - p.K_lt * F_y_LF) \
                  - F_z_RF * (-p.R_w * torch.sin(state[..., 13]) + 0.5 * p.T_f * torch.cos(state[..., 13]) + p.K_lt * F_y_RF) \
                  - ((F_y_LF + F_y_RF) * torch.cos(state[..., 2]) + (F_x_LF + F_x_RF) * torch.sin(state[..., 2])) * (p.R_w - state[..., 16])

        sumL_ur = 0.5 * F_SRR * p.T_r - 0.5 * F_SLR * p.T_r - F_RAR * (p.h_rar - p.R_w) \
                  + F_z_LR * (p.R_w * torch.sin(state[..., 18]) + 0.5 * p.T_r * torch.cos(state[..., 18]) - p.K_lt * F_y_LR) \
                  - F_z_RR * (-p.R_w * torch.sin(state[..., 18]) + 0.5 * p.T_r * torch.cos(state[..., 18]) + p.K_lt * F_y_RR) \
                  - (F_y_LR + F_y_RR) * (p.R_w - state[..., 21])

        sumZ_uf = F_z_LF + F_z_RF + F_RAF * torch.sin(state[..., 6]) - (F_SLF + F_SRF) * torch.cos(state[..., 6])

        sumZ_ur = F_z_LR + F_z_RR + F_RAR * torch.sin(state[..., 6]) - (F_SLR + F_SRR) * torch.cos(state[..., 6])

        sumY_uf = (F_y_LF + F_y_RF) * torch.cos(state[..., 2]) + (F_x_LF + F_x_RF) * torch.sin(state[..., 2]) \
                  - F_RAF * torch.cos(state[..., 6]) - (F_SLF + F_SRF) * torch.sin(state[..., 6])

        sumY_ur = (F_y_LR + F_y_RR) \
                  - F_RAR * torch.cos(state[..., 6]) - (F_SLR + F_SRR) * torch.sin(state[..., 6])

        # dynamics common with single-track model

        # switch to kinematic model for small velocities
        dx = torch.cos(beta + state[..., 4]) * vel
        dy = torch.sin(beta + state[..., 4]) * vel
        dalpha = action[..., 0]
        dv = 1 / p.m * sumX + state[..., 5] * state[..., 10]
        dv[inds] = action[inds, 1]
        dpsi = state[..., 5]
        dpsi[inds] = state[inds, 3] * torch.cos(beta[inds]) * torch.tan(state[inds, 2]) / p.l_wb
        d_beta = (p.b * action[..., 0]) / (
                    p.l_wb * torch.cos(state[..., 2]) ** 2 * (1 + (torch.tan(state[..., 2]) ** 2 * p.b / p.l_wb) ** 2))
        ddpsi = 1 / p.l_wb * (action[..., 1] * torch.cos(state[..., 6]) * torch.tan(state[..., 2]) -
                            state[..., 3] * torch.sin(state[..., 6]) * d_beta * torch.tan(state[..., 2]) +
                            state[..., 3] * torch.cos(state[..., 6]) * action[..., 0] / torch.cos(state[..., 2]) ** 2)
        _inds = torch.where(torch.abs(state[:, 3]) >= 0.1)[0]
        ddpsi[_inds] = 1 / (p.I_z - (p.I_xz_s) ** 2 / p.I_Phi_s) * (sumN[_inds] + p.I_xz_s / p.I_Phi_s * sumL[_inds])

        f = [dx, dy, dalpha, dv, dpsi, ddpsi]
        # remaining sprung mass dynamics
        f.append(state[..., 7])
        f.append(1 / (p.I_Phi_s - (p.I_xz_s) ** 2 / p.I_z) * (p.I_xz_s / p.I_z * sumN + sumL))
        f.append(state[..., 9])
        f.append(1 / p.I_y_s * sumM_s)
        f.append(1 / p.m_s * sumY_s - state[..., 5] * state[..., 3])
        f.append(state[..., 12])
        f.append(p.g - 1 / p.m_s * sumZ_s)

        # unsprung mass dynamics (front)
        f.append(state[..., 14])
        f.append(1 / p.I_uf * sumL_uf)
        f.append(1 / p.m_uf * sumY_uf - state[..., 5] * state[..., 3])
        f.append(state[..., 17])
        f.append(p.g - 1 / p.m_uf * sumZ_uf)

        # unsprung mass dynamics (rear)
        f.append(state[..., 19])
        f.append(1 / p.I_ur * sumL_ur)
        f.append(1 / p.m_ur * sumY_ur - state[..., 5] * state[..., 3])
        f.append(state[..., 22])
        f.append(p.g - 1 / p.m_ur * sumZ_ur)

        # convert acceleration input to brake and engine torque
        T_B = p.m * p.R_w * action[..., 1]
        T_E = p.m * p.R_w * action[..., 1]
        T_B[torch.where(action[:, 1] > 0)[0]] = 0
        T_E[torch.where(action[:, 1] <= 0)[0]] = 0

        # wheel dynamics (p.T  new parameter for torque splitting)
        f.append(1 / p.I_y_w * (-p.R_w * F_x_LF + 0.5 * p.T_sb * T_B + 0.5 * p.T_se * T_E))
        f.append(1 / p.I_y_w * (-p.R_w * F_x_RF + 0.5 * p.T_sb * T_B + 0.5 * p.T_se * T_E))
        f.append(1 / p.I_y_w * (-p.R_w * F_x_LR + 0.5 * (1 - p.T_sb) * T_B + 0.5 * (1 - p.T_se) * T_E))
        f.append(1 / p.I_y_w * (-p.R_w * F_x_RR + 0.5 * (1 - p.T_sb) * T_B + 0.5 * (1 - p.T_se) * T_E))

        # negative wheel spin forbidden

        for iState in range(23, 27):
            state[torch.where(state[..., iState] < 0)[0]] = 0
            f[iState] = torch.zeros(f[iState].shape[0], device=f[iState].device)

        # compliant joint equations
        f.append(dot_delta_y_f)
        f.append(dot_delta_y_r)

        new_states = []
        for i in range(len(f)):
            new_states.append(state[..., i] + f[i].clone().detach().requires_grad_(state.requires_grad) * dt)
        next_state = torch.stack(new_states, dim=-1)

        return next_state

    def loss(self, state, action, ref, dims, factors, eval_act=True, mean=True):
        Rd = [0.1, 0.1]  # input difference cost matrix
        Q = [1, 1, 10, 100, 100, 100]
        loss = 0
        for i in range(len(Q)):
            loss += Q[i] * (state[..., dims[i]] - ref[..., i])**2
        if eval_act:
            for i in range(len(Rd)):
                loss += Rd[i] * (action[..., i]**2)

        return torch.mean(loss) if mean else loss
