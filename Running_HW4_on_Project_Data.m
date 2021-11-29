
clear all; close all; format compact; clc
rootFolder = cd;

prompt = 'Use adaptive method? Yes = 1 or No = 0\n';
Adaptive = input(prompt);
prompt = 'Case Number\n';
CaseNo = input(prompt);

data_padding = 20;

cd(strcat(rootFolder,'\Project_2_Data'))
data = readmatrix('INR21700_M50T_T23_OCV_W8.xlsx');
t = data(:,2);
Voc_vs_SOC(:,2) = data(:,3);
I = -data(:,4);
Q   = cumtrapz(t,I)/3600;
Qn_OCV  = Q(end);
Voc_vs_SOC(:,1) = ( 1 - Q/Qn_OCV );
SOC_map = 0:0.0001:1;
OCV_map = interp1(Voc_vs_SOC(:,1),Voc_vs_SOC(:,2),SOC_map);

figure(); set(gcf,'color','w'); hold on;
plot(100*Voc_vs_SOC(:,1),Voc_vs_SOC(:,2),'DisplayName','All Points');
plot(100*SOC_map,OCV_map,'DisplayName','Reduced Points');
title('OCV vs. SOC');
xlabel('SOC (%)');
ylabel('OCV (Voltage)');
legend('Location','Best');


data_capacity  = readmatrix('Capacity_Values.xlsx');
n_vec  = data_capacity(:,1);
Qn_vec = data_capacity(:,2);

if CaseNo==1 | CaseNo==2 | CaseNo==3 | CaseNo==4
    cd(strcat(rootFolder,'\Project_2_Data','\HPPC'))
    start_idx = 14476;
else
    cd(strcat(rootFolder,'\Project_2_Data','\UDDS'))
end
switch CaseNo
    case 1
        data = readmatrix('INR21700_M50T_T23_HPPC_N0_W8.xlsx');
        n = n_vec(1);
        Qn = Qn_vec(1);
        fileName = 'HPPCresult_N0_HW4_EKF';
        
%         t = data(:,2);
%         V_expt = data(:,3);
%         I_expt = -data(:,4);
%         SOC_CC = 0 - (cumtrapz(t, I_expt)/3600)/Qn;
%         
%         % ---- Charging ----
%         %parsing data
%         [start_chg_idxs, end_chg_idxs] = find_charge_idxs(I_expt,data_padding);
%         %graphically estimating parameters
%         [soc_chg,R0_chg,R1_chg,C1_chg,R2_chg,C2_chg] = ...
%             estimate_parameters_graphically(start_chg_idxs,end_chg_idxs,t,V_expt,I_expt,SOC_CC,data_padding);
%         %forming upper and lower bounds on parameter values
%         ECM_params_graphical_chg = [R0_chg,R1_chg,C1_chg,R2_chg,C2_chg];
%         ub_chg = 1.2*ECM_params_graphical_chg;
%         lb_chg = 0.8*ECM_params_graphical_chg;
%         %running genertic algorithm with bounds
%         [R0_chg,R1_chg,C1_chg,R2_chg,C2_chg,prmse_chg] = ...
%             estimate_parameters_ga(start_chg_idxs, end_chg_idxs,t,V_expt,I_expt,SOC_CC,SOC_map,OCV_map,ub_chg,lb_chg);
%         ECM_params_ga_chg = [R0_chg,R1_chg,C1_chg,R2_chg,C2_chg];
% 
%         % ---- Discharging ----
%         %parsing data
%         [start_dischg_idxs, end_dischg_idxs] = find_discharge_idxs(I_expt,data_padding);
%         %graphically estimating parameters
%         [soc_dischg,R0_dischg,R1_dischg,C1_dischg,R2_dischg,C2_dischg] = ...
%             estimate_parameters_graphically(start_dischg_idxs, end_dischg_idxs,t,V_expt,I_expt,SOC_CC,data_padding);
%         %forming upper and lower bounds on parameter values
%         ECM_params_graphical_dischg = [R0_dischg,R1_dischg,C1_dischg,R2_dischg,C2_dischg];
%         ub_disch = 1.2*ECM_params_graphical_dischg;
%         lb_disch = 0.8*ECM_params_graphical_dischg;
%         %running genertic algorithm with bounds
%         [R0_dischg,R1_dischg,C1_dischg,R2_dischg,C2_dischg,prmse_dischg] = ...
%             estimate_parameters_ga(start_dischg_idxs, end_dischg_idxs,t,V_expt,I_expt,SOC_CC,SOC_map,OCV_map,ub_disch,lb_disch);
%         ECM_params_ga_dischg = [R0_dischg,R1_dischg,C1_dischg,R2_dischg,C2_dischg];
%         
%         clear t V_expt I_expt SOC_CC
%         save('parameters.mat')
        load('parameters.mat')
    case 2
        data = readmatrix('INR21700_M50T_T23_HPPC_N75_W8.xlsx');
        n = n_vec(1);
        Qn = Qn_vec(2);
        fileName = 'HPPCresult_N75_HW4_EKF';
        load('parameters.mat')
    case 3
        data = readmatrix('INR21700_M50T_T23_HPPC_N125_W8.xlsx');
        n = n_vec(1);
        Qn = Qn_vec(3);
        fileName = 'HPPCresult_N125_HW4_EKF';
        load('parameters.mat')
    case 4
        data = readmatrix('INR21700_M50T_T23_HPPC_N200_W8.xlsx');
        n = n_vec(1);
        Qn = Qn_vec(4);
        fileName = 'HPPCresult_N200_HW4_EKF';
        load('parameters.mat')
    case 5
        data = readmatrix('INR21700_M50T_T23_UDDS_W8_N1.xlsx');
        n = 1;
        Qn = max(cumtrapz(data(:,2),data(:,4))/3600);
        start_idx = 13012;
        fileName = 'UDDSresult_N1_HW4_EKF';
    case 6
        data = readmatrix('INR21700_M50T_T23_UDDS_W8_N201.xlsx');
        n = 201;
        Qn = max(cumtrapz(data(:,2),data(:,4))/3600);
        start_idx = 11196;
        fileName = 'UDDSresult_N201_HW4_EKF';
end

if Adaptive == 1
    fileName = strcat(fileName,'_adaptive')
end

t = data(:,2);
V_expt = data(:,3);
I_expt = -data(:,4);
SOC_CC = 0 - (cumtrapz(t, I_expt)/3600)/Qn;

t = t(start_idx:end);
V_expt = V_expt(start_idx:end);
I_expt = I_expt(start_idx:end);
SOC_CC = SOC_CC(start_idx:end);

N = length(V_expt);
W = 10;

R0_init = mean([ECM_params_ga_dischg(:,1); ECM_params_ga_chg(:,1)]);
P_t = [0.1 0 0 ; 0 0.001 0 ; 0 0 0.001];
x_t(:,1) = [SOC_CC(1); 0; 0];
R = 8.432e-4;
Q =  diag([1000*R,0.1*R,0.01*R]);
% R = 131.8108;
% Q = [493.7918         0         0         0;
%              0    0.0031         0         0;
%              0         0    0.0018         0;
%              0         0         0    0.6748];
deltaSOC = 0.01;
max_growth = 0.05;


for i = 2:N
    % Predict
    dt = t(i)-t(i-1);
    if I_expt(i-1) < 0
        R1 = interp1(soc_chg, R1_chg, x_t(1,i-1), 'linear','extrap');
        R2 = interp1(soc_chg, R2_chg, x_t(1,i-1), 'linear','extrap');
        C1 = interp1(soc_chg, C1_chg, x_t(1,i-1), 'linear','extrap');
        C2 = interp1(soc_chg, C2_chg, x_t(1,i-1), 'linear','extrap');
        dR1 = derivative(soc_chg, R1_chg, x_t(1,i-1),deltaSOC);
        dR2 = derivative(soc_chg, R2_chg, x_t(1,i-1),deltaSOC);
        dC1 = derivative(soc_chg, C1_chg, x_t(1,i-1),deltaSOC);
        dC2 = derivative(soc_chg, C2_chg, x_t(1,i-1),deltaSOC);
    else
        R1 = interp1(soc_dischg, R1_dischg, x_t(1,i-1), 'linear','extrap');
        R2 = interp1(soc_dischg, R2_dischg, x_t(1,i-1), 'linear','extrap');
        C1 = interp1(soc_dischg, C1_dischg, x_t(1,i-1), 'linear','extrap');
        C2 = interp1(soc_dischg, C2_dischg, x_t(1,i-1), 'linear','extrap');
        dR1 = derivative(soc_dischg, R1_dischg, x_t(1,i-1),deltaSOC);
        dR2 = derivative(soc_dischg, R2_dischg, x_t(1,i-1),deltaSOC);
        dC1 = derivative(soc_dischg, C1_dischg, x_t(1,i-1),deltaSOC);
        dC2 = derivative(soc_dischg, C2_dischg, x_t(1,i-1),deltaSOC);
    end
    
    
    A = [0, 0, 0; ...
         ( x_t(2,i-1)*(R1*dC1 + C1*dR1)/(R1*C1)^2 - I_expt(i-1)*dC1/C1^2 ), ( -1/(R1*C1) ), 0;
         ( x_t(3,i-1)*(R2*dC2 + C2*dR2)/(R2*C2)^2 - I_expt(i-1)*dC2/C2^2 ), 0,              ( -1/(R2*C2) )];
    B = [-1/(3600*Qn); 1/C1; 1/C2];

    x_tp(:,i) = x_t(:,i-1) + dt*[-I_expt(i-1)/(3600*Qn);...
                                -x_t(2,i-1)/(R1*C1) + I_expt(i-1)/C1;...
                                -x_t(3,i-1)/(R2*C2) + I_expt(i-1)/C2];
    x_tp(1,i) = saturate(x_tp(1,i),0,1);
    P_tp = A*P_t*A' + Q;
    
    % Correct
    if I_expt(i) < 0
        R0 = interp1(soc_chg, R0_chg, x_tp(1,i), 'linear','extrap');
        dR0 = derivative(soc_chg, R0_chg, x_tp(1,i),deltaSOC);
    else
        R0 = interp1(soc_dischg, R0_dischg, x_tp(1,i), 'linear','extrap');
        dR0 = derivative(soc_dischg, R0_dischg, x_tp(1,i),deltaSOC);
    end
    dV_dSOC = derivative(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), ...
                                x_tp(1,i),deltaSOC);
    C = [(dV_dSOC - I_expt(i)*dR0), -1, -1];

    L(:,i) = P_tp * C' * inv(C*P_tp*C' + R);
    Voc = interp1(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), x_tp(1,i),'linear','extrap');
    V = Voc - x_tp(2,i) - x_tp(3,i) - I_expt(i)*R0;
    x_t(:,i) = x_tp(:,i) + L(:,i)*(V_expt(i) - V);
    x_t(1,i) = saturate(x_t(1,i),0,1);
    P_t = (eye(length(A)) - L(:,i)*C)*P_tp;
    
    if I_expt(i) < 0
        R0 = interp1(soc_chg, R0_chg, x_t(1,i), 'linear','extrap');
    else
        R0 = interp1(soc_dischg, R0_dischg, x_t(1,i), 'linear','extrap');
    end
    Voc = interp1(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), x_t(1,i), 'linear','extrap');
    Vb(i) = Voc - x_t(2,i) - x_t(3,i) - I_expt(i)*R0;
    R0_with_time(i) = R0;

    if Adaptive
        % Adapt
        d(i) = V_expt(i)-V;
        if length(Vb) < W
            D = 1/W*sum(d*d');
            delta_x = x_t-x_tp;
            L_adapt = delta_x/d;
            Q = L_adapt*D*L_adapt';
        else
            D = 1/W*sum( d(i-W+1:i) * d(i-W+1:i)' );
            delta_x = x_t(:,i-W+1:i) - x_tp(:,i-W+1:i);
            L_adapt = delta_x/d(i-W+1:i);
            Q = L_adapt*D*L_adapt';
        end
    end
end


prmse_SOC = calc_percent_rmse(SOC_CC,x_t(1,:));
prmse_V = calc_percent_rmse(V_expt,Vb);
disp(['RMS Error in SOC = ' num2str(prmse_SOC) '%']);
disp(['RMS Error in voltage = ' num2str(prmse_V) '%']);

cd(rootFolder)
save(fileName)

figure(); set(gcf,'color','w'); hold on;
plot(t,V_expt);
plot(t(1:length(Vb)),Vb)
xlabel('Time [s]'); ylabel('Voltage [V]')
legend('Experimental','Kalman')

figure(); set(gcf,'color','w'); hold on;
plot(t,SOC_CC*100);
plot(t(1:length(Vb)),x_t(1,:)*100)
xlabel('Time [s]'); ylabel('SOC [%]')
legend('Experimental','Kalman')

figure(); set(gcf,'color','w'); hold on;
plot(t(1:length(Vb)),R0_with_time)
xlabel('Time [s]'); ylabel('R0')

load handel
sound(y,Fs)



function result = derivative(x,y,x_point,delta_x)
    y1 = interp1(x, y, x_point-delta_x, 'linear','extrap');
    y2 = interp1(x, y, x_point+delta_x, 'linear','extrap');
    result = (y2-y1)/(2*delta_x);
end

function y = saturate(x,lb,ub)
    if x < lb 
        y = lb;
    elseif x > ub
        y = ub;
    else
        y = x;
    end
end

function prmse = calc_percent_rmse(x_actual,x_corrupted)
    N = length(x_actual);
    prmse = sqrt((1/N)*sum((reshape(x_corrupted,[],1) - reshape(x_actual,[],1)).^2))*(100*N/sum(x_actual));
end


    