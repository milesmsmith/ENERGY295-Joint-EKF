clear all; close all; format compact; clc

load('HW4_batt_params.mat');
load('HW4_expt_data.mat');

N = length(V_expt);
t = [0:dt:(N-1)]';

R0 = mean([R0_chg,R0_dischg]);
R1 = mean([R1_chg,R1_dischg]);
C1 = mean([C1_chg,C1_dischg]);
R2 = mean([R2_chg,R2_dischg]);
C2 = mean([C2_chg,C2_dischg]);
tau1 = R1*C1;
tau2 = R2*C2;

P_t = [0.1 0 0; 0 0.001 0; 0 0 0.001];
x_t(:,1) = [0; 0; 0];
R = 8.432e-4;
Q = diag([1000*R 0.1*R 0.01*R]);
deltaSOC = 0.01;

SOC_CC = SOC_init - (cumtrapz(t, I_expt)/3600)/capacity;

for i = 2:N
    Ad = [1 0 0; 0 exp(-dt/tau1) 0; 0 0 exp(-dt/tau2)];
    Bd = [-dt/3600/capacity; R1*(1-exp(-dt/tau1)); R2*(1-exp(-dt/tau2))];

    V1 = interp1(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), x_t(1,i-1)-deltaSOC, 'linear','extrap');
    V2 = interp1(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), x_t(1,i-1)+deltaSOC, 'linear','extrap');
    Cd = [(V2-V1)/(2*deltaSOC) -1 -1];
    
    % Predict
    x_tp(:,i) = Ad*x_t(:,i-1) + Bd*I_expt(i-1);
    P_tp = Ad*P_t*Ad' + Q;
    
    % Correct
    L = P_tp * Cd' * inv(Cd*P_tp*Cd' + R);
    Voc = interp1(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), x_tp(1,i), 'linear','extrap');
    V = Voc - x_tp(2,i) - x_tp(3,i) - I_expt(i)*R0;
    x_t(:,i) = x_tp(:,i) + L*(V_expt(i) - V);
    P_t = (eye(length(Ad)) - L*Cd)*P_tp;
    
    Voc = interp1(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), x_t(1,i), 'linear','extrap');
    Vb(i) = Voc - x_t(2,i) - x_t(2,i) - I_expt(i)*R0;
end

figure(1); set(gcf,'color','w'); hold on;
plot(t,V_expt);
plot(t(2:end),Vb(2:end))
xlabel('Time [s]'); ylabel('Voltage [V]')
legend('Experimental','Kalman')

figure(2); set(gcf,'color','w'); hold on;
plot(t,SOC_CC*100);
plot(t,x_t(1,:)*100)
xlabel('Time [s]'); ylabel('SOC [%]')
legend('Experimental','Kalman')