clear all; close all; format compact; clc

load('HW4_batt_params.mat');
load('HW4_expt_data.mat');

N = length(V_expt);
W = 10;
t = [0:dt:(N-1)]';

% R0 = mean([R0_chg,R0_dischg]);
% R1 = mean([R1_chg,R1_dischg]);
% C1 = mean([C1_chg,C1_dischg]);
% R2 = mean([R2_chg,R2_dischg]);
% C2 = mean([C2_chg,C2_dischg]);
% tau1 = R1*C1;
% tau2 = R2*C2;

% Initialize
P_t = [0.1 0 0; 0 0.001 0; 0 0 0.001];
x_t(:,1) = [0; 0; 0];
R = 8.432e-4;
Q = diag([1000*R 0.1*R 0.01*R]);
deltaSOC = 0.01;

SOC_CC = SOC_init - (cumtrapz(t, I_expt)/3600)/capacity;

for i = 2:N
    if I_expt(i-1)<0
        R0 = interp1(soc_chg, R0_chg, x_t(1,i-1), 'linear','extrap');
        R1 = interp1(soc_chg, R1_chg, x_t(1,i-1), 'linear','extrap');
        R2 = interp1(soc_chg, R2_chg, x_t(1,i-1), 'linear','extrap');
        C1 = interp1(soc_chg, C1_chg, x_t(1,i-1), 'linear','extrap');
        C2 = interp1(soc_chg, C2_chg, x_t(1,i-1), 'linear','extrap');
        dR0 = derivative(soc_chg, R0_chg, x_t(1,i-1),deltaSOC);
        dR1 = derivative(soc_chg, R1_chg, x_t(1,i-1),deltaSOC);
        dR2 = derivative(soc_chg, R2_chg, x_t(1,i-1),deltaSOC);
        dC1 = derivative(soc_chg, C1_chg, x_t(1,i-1),deltaSOC);
        dC2 = derivative(soc_chg, C2_chg, x_t(1,i-1),deltaSOC);
    else
        if I_expt(i-1)>=4
            R0 = interp1(soc_chg, R0_dischg_4A, x_t(1,i-1), 'linear','extrap');
            dR0 = derivative(soc_dischg, R0_dischg_4A, x_t(1,i-1),deltaSOC);
        else
            R0 = interp1(soc_chg, R0_dischg, x_t(1,i-1), 'linear','extrap');
            dR0 = derivative(soc_dischg, R0_dischg, x_t(1,i-1),deltaSOC);
        end
        R1 = interp1(soc_chg, R1_dischg, x_t(1,i-1), 'linear','extrap');
        R2 = interp1(soc_chg, R2_dischg, x_t(1,i-1), 'linear','extrap');
        C1 = interp1(soc_chg, C1_dischg, x_t(1,i-1), 'linear','extrap');
        C2 = interp1(soc_chg, C2_dischg, x_t(1,i-1), 'linear','extrap');
        dR1 = derivative(soc_dischg, R1_dischg, x_t(1,i-1),deltaSOC);
        dR2 = derivative(soc_dischg, R2_dischg, x_t(1,i-1),deltaSOC);
        dC1 = derivative(soc_dischg, C1_dischg, x_t(1,i-1),deltaSOC);
        dC2 = derivative(soc_dischg, C2_dischg, x_t(1,i-1),deltaSOC);
    end
    
    dV_dSOCV = derivative(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), ...
                                x_t(1,i-1),deltaSOC);
    A = [0 0 0; ...
         ( x_t(2,i-1)*(R1*dC1 + C1*dR1)/(R1*C1)^2 - I_expt(i-1)*dC1/C1^2 ) ( -1/(R1*C1) ) 0;
         ( x_t(3,i-1)*(R2*dC2 + C2*dR2)/(R2*C2)^2 - I_expt(i-1)*dC2/C2^2 ) 0              ( -1/(R2*C2) )];
    B = [-1/(3600*capacity); 1/C1; 1/C2];
    C = [(dV_dSOCV - I_expt(i-1)*dR0) -1 -1];

    % Predict
    x_tp(:,i) = x_t(:,i-1) + dt*[-I_expt(i-1)/(3600*capacity);...
                                -x_t(2,i-1)/(R1*C1) + I_expt(i-1)/C1;...
                                -x_t(3,i-1)/(R2*C2) + I_expt(i-1)/C2];
    P_tp = A*P_t*A' + Q;
    
    % Correct
    L = P_tp * C' * inv(C*P_tp*C' + R);
    Voc = interp1(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), x_tp(1,i),'linear','extrap');
    V = Voc - x_tp(2,i) - x_tp(3,i) - I_expt(i)*R0;
    x_t(:,i) = x_tp(:,i) + L*(V_expt(i) - V);
    P_t = (eye(length(A)) - L*C)*P_tp;
    
    Voc = interp1(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), x_t(1,i), 'linear','extrap');
    Vb(i) = Voc - x_t(2,i) - x_t(3,i) - I_expt(i)*R0;
    
    % Adapt
    d(i) = V_expt(i)-V;
    if length(Vb)<10
        D = 1/W*sum(d*d');
        delta_x = x_t-x_tp;
        L = delta_x/d;
        Q = L*D*L';
    else
        D = 1/W*sum( d(i-W+1:i) * d(i-W+1:i)' );
        delta_x = x_t(:,i-W+1:i) - x_tp(:,i-W+1:i);
        L = delta_x/d(i-W+1:i);
        Q = L*D*L';
    end
end

figure(1); set(gcf,'color','w'); hold on;
plot(t,V_expt);
plot(t,Vb)
xlabel('Time [s]'); ylabel('Voltage [V]')
legend('Experimental','Kalman')

figure(2); set(gcf,'color','w'); hold on;
plot(t,SOC_CC*100);
plot(t,x_t(1,:)*100)
xlabel('Time [s]'); ylabel('SOC [%]')
legend('Experimental','Kalman')

function result = derivative(x,y,x_point,delta_x)
    y1 = interp1(x, y, x_point-delta_x, 'linear','extrap');
    y2 = interp1(x, y, x_point+delta_x, 'linear','extrap');
    result = (y2-y1)/(2*delta_x);
end