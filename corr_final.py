import numpy as np
from math import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import control
import control.matlab


class Model:
    #Вектор состояния
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z0 = np.array([x0[0], x0[1]])
    xd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    #Шаг интегрирования
    dt = 0.005
    tt = np.array([0, dt])
    T = 50
    TT = np.arange(0, T + dt, dt)
    u0 = np.array([0.0])

    A = np.array([[0, 1],
                  [-5.42056, -0.0561]])

    B = np.array([[0],
                  [2.2318]])

    Q0_lqr = np.diag(np.array([1, 2]))
    R0_lqr = np.diag(np.array([0.01]))
    xx = np.array(x0)
    uu = np.array(u0)
    H = 10
    al = -1
    p0 = 0

    def F(self, VX, t, U):
        TetaV, OmegaV = VX[0], VX[1]
        TetaH, OmegaH = VX[2], VX[3]
        a1 = 0.0135  # Main Rotor Coefficient
        b1 = 0.0924  # Main Rotor Coefficient
        a2 = 0.02  # Tail Rotor Coefficient
        b2 = 0.09  # Tail Rotor Coefficient
        B1tv = 0.006  # Friction Momentum
        B1th = 0.1  # Friction Momentum
        B2t = 0.0326 / 2  # Friction Momentum
        Mg = 0.32  # Moment of Gravity
        I1 = 0.068  # Pitch Moment of Inertia
        I2 = 0.02  # Yaw Moment of Inertia
        Kgy = 0.05  # Gyroscopic Momentum
        Tp = 2  # Cross Reaction Momentum Parameter
        T0 = 3.5  # Cross Reaction Momentum Parameter
        Kc = -0.2  # Cross Reaction Momentum Gain
        T10 = 1  # Main Rotor Denomnatior
        T11 = 1.1  # Main Rotor Denomnatior
        T20 = 1  # Tail Rotor Denominator
        T21 = 1  # Tail Rotor Denominator
        K1 = 1.1  # Main rotor gain
        K2 = 0.8  # Tail rotor gain

        val = 0.0326 / 2
        U=np.squeeze(np.asarray(U))

        # Уравнения вертикальной и горизонтальной системы
        tau1 = U
        tau2 = 0

        # Нелинейные статические характеристики роторов
        M1 = a1 * (tau1 ** 2) + b1 * tau1
        M2 = a2 * (tau2 ** 2) + b2 * tau2

        # Моменты вызванные трением
        MBtv = B1tv * OmegaV - val * sin(2 * TetaV) * (OmegaV ** 2)
        MBth = B1th * OmegaH - val * sin(2 * TetaH) * (OmegaH ** 2)
        # Момент силы тяжести
        MFG = Mg * sin(TetaV)
        # Гироскопический момент
        MG = Kgy * M1 * OmegaH * cos(TetaV)
        # Момент перекрестной реакции
        Mr = Kc * (T0 + 1) / (Tp + 1)

        dX1 = OmegaV

        dX2 = (a1 * (tau1 ** 2) + b1 * tau1 - Mg * sin(TetaV) - B1tv * OmegaV + val * sin(2 * TetaV) *
               (OmegaV ** 2) - Kgy * a1 * cos(TetaV) * OmegaH * (tau1 ** 2) - Kgy * b1 * cos(TetaV) * OmegaH * tau1)/I1 + 0.1

        dX3 = OmegaH

        dX4 = (a2 * (tau2 ** 2) + b2 * tau2 - B1th * OmegaH - (Kc / Tp - Kc * Tp / (Tp ** 2)) * Mr -
              -(Kc * T0 / Tp) * (a1 * (tau1 ** 2) + b1 * tau1)) / I2

        dX5 = -Mr / Tp + a1 * (tau1 ** 2) + b1 * tau1

        dX6 = -T10 * tau1 / T11 + K1 * U / T11

        dX7 = -T20 * tau2 / T21 + K2 * U / T21
        #print(type(np.array([dX1, dX2])))
        return np.array([dX1, dX2, dX3, dX4, dX5, dX6, dX7])

    def make_klqr(self):
        Q0 = self.Q0_lqr
        R0 = self.R0_lqr
        self.klqr = -control.lqr(self.A, self.B, Q0, R0)[0]
        ao = self.A[1,0]
        aw = self.A[1,1]
        b  = self.B[1,0]
        ko = self.klqr[0,0]
        kw = self.klqr[0,1]
        h = self.H
        self.bt = (aw*h + b*kw*h - h + ao + b*ko)/b
        self.xd = self.xd*(ao+b*ko)/(ko*b)
        print(f"K0: {self.bt}")
        print(f"klqr: {self.klqr}")

    def lqr(self, x):
        x = np.array([x[0], x[1]])
        xd = np.array([self.xd[0], self.xd[1]])
        u = self.klqr @ (x - xd)
        return u

    def ctrl(self, x, p):
        u = self.lqr(x) #+ p
        return u

    def rp(self, in_, t, u):
        print(type(in_), in_)
        x = in_.reshape((2, 1))

       
        aa = self.A @ x
        bb = self.B * u

        dx = aa + bb  + 1 * np.array([[0, 0.1]]).T
        print(type(np.squeeze(np.asarray(dx))))
        return np.squeeze(np.asarray(dx))
        #return F(x, u)
    
    def rp_obsv(self, in_, t, u, x):
        x = np.array([x[0], x[1]])
        z = in_.reshape((2, 1))
        x = x.reshape((2, 1))
       
        aa = self.A @ z
        bb = self.B * u
        hh = self.H * (x[0] - z[0])
        bb = bb.reshape(2,1)

        dz = aa + bb + hh
        dz = np.squeeze(np.asarray(dz))
        return dz

    def rp_corr(self, in_, t, z, x):
        aa = -in_
        bb = self.bt * (x[0] - z[0])
        dp = aa + bb
        return dp

    def state(self, u, x0):
        x = odeint(self.F, x0, self.tt, args=(u,))[-1, :]
        return x
    
    def obsv(self, u, x, z0):
        z = odeint(self.rp_obsv, z0, self.tt, args=(u,x))[-1, :]
        return z

    def corr(self, x, z, p0):
        p = odeint(self.rp_corr, p0, self.tt, args=(z,x))[-1, :]
        return p

    def step(self, x0, z0, p0):
        u = self.ctrl(z0, p0)
        x = self.state(u, x0)
        z = self.obsv(u, x0, z0)
        p = self.corr(x, z, p0)
        return (x, u, z, p)

    def main_cycle(self):
        x = self.x0
        u = self.u0
        z = self.z0
        p = self.p0
        for t in self.TT[1:]:
            (x, u, z, p) = self.step(x, z, p)
            self.xx = np.vstack((self.xx, x))
            self.uu = np.vstack((self.uu, u))

    def __init__(self):
        self.make_klqr()
        pass

m = Model()
m.main_cycle()

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(m.TT, m.xx[:, 0], label='pitch')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(m.TT, m.xx[:, 1], label='w')
plt.legend()
plt.grid()

plt.figure(2)
plt.plot(m.TT, m.uu, label='ctrl')
plt.legend()
plt.grid()

plt.show()
