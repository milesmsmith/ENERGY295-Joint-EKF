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


cd(strcat(rootFolder,'\Project_2_Data','\HPPC'))
load('parameters.mat')
if CaseNo==1 | CaseNo==2 | CaseNo==3 | CaseNo==4
    start_idx = 14476;
else
    cd(strcat(rootFolder,'\Project_2_Data','\UDDS'))
end
switch CaseNo
    case 1
        data = readmatrix('INR21700_M50T_T23_HPPC_N0_W8.xlsx');
        n = n_vec(1);
        Qn = Qn_vec(1);
        fileName = 'HPPCresult_N0';
        
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
%         clearvars -except ECM_params_ga_chg R0_chg R1_chg C1_chg R2_chg...
%             C2_chg ECM_params_ga_dischg R0_dischg R1_dischg C1_dischg...
%             R2_dischg C2_dischg soc_chg soc_dischg
%         save('parameters.mat')

    case 2
        data = readmatrix('INR21700_M50T_T23_HPPC_N75_W8.xlsx');
        n = n_vec(1);
        Qn = Qn_vec(2);
        fileName = 'HPPCresult_N75';
        load('parameters.mat')
    case 3
        data = readmatrix('INR21700_M50T_T23_HPPC_N125_W8.xlsx');
        n = n_vec(1);
        Qn = Qn_vec(3);
        fileName = 'HPPCresult_N125';
    case 4
        data = readmatrix('INR21700_M50T_T23_HPPC_N200_W8.xlsx');
        n = n_vec(1);
        Qn = Qn_vec(4);
        fileName = 'HPPCresult_N200';
    case 5
        data = readmatrix('INR21700_M50T_T23_UDDS_W8_N1.xlsx');
        n = 1;
        Qn = max(cumtrapz(data(:,2),data(:,4))/3600);
        start_idx = 13012;
        fileName = 'UDDSresult_N1';
    case 6
        data = readmatrix('INR21700_M50T_T23_UDDS_W8_N201.xlsx');
        n = 201;
        Qn = max(cumtrapz(data(:,2),data(:,4))/3600);
        start_idx = 11196;
        fileName = 'UDDSresult_N201';
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
P_t = [0.1 0 0 0; 0 0.001 0 0; 0 0 0.001 0; 0 0 0 0.001];
x_t(:,1) = [SOC_CC(1); 0; 0; R0_init];
R = 8.432e-4;
Q =  diag([1000*R,0.1*R,0.01*R,4.7796]);
% R = 131.8108;
% Q = [493.7918         0         0         0;
%              0    0.0031         0         0;
%              0         0    0.0018         0;
%              0         0         0    0.6748];
deltaSOC = 0.01;
max_growth = 0.05;


for i = 2:N
    if I_expt(i-1)<0
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
    
    A = [0                                                                ,  0             ,  0             ,  0; ...
         ( x_t(2,i-1)*(R1*dC1 + C1*dR1)/(R1*C1)^2 - I_expt(i-1)*dC1/C1^2 ),  ( -1/(R1*C1) ),  0             ,  0;
         ( x_t(3,i-1)*(R2*dC2 + C2*dR2)/(R2*C2)^2 - I_expt(i-1)*dC2/C2^2 ),  0             ,  ( -1/(R2*C2) ),  0;
         0                                                                ,  0             ,  0             ,  1];
    B = [-1/(3600*Qn); 1/C1; 1/C2; 0];

    % Predict
    dt = t(i)-t(i-1);
    x_tp(:,i) = x_t(:,i-1) + dt*[-I_expt(i-1)/(3600*Qn);...
                                -x_t(2,i-1)/(R1*C1) + I_expt(i-1)/C1;...
                                -x_t(3,i-1)/(R2*C2) + I_expt(i-1)/C2;...
                                0];
    x_tp(1,i) = saturate(x_tp(1,i),0,1);
    x_tp(4,i) = growth(x_tp(4,i), x_t(4,i-1), max_growth);
    P_tp = A*P_t*A' + Q;
    
    % Correct
    dVoc_dSOCV = derivative(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), ...
                                x_tp(1,i),deltaSOC);
    C = [dVoc_dSOCV, -1, -1, -I_expt(i)];
    L = P_tp * C' * inv(C*P_tp*C' + R);
    Voc = interp1(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), x_tp(1,i),'linear','extrap');
    V = Voc - x_tp(2,i) - x_tp(3,i) - I_expt(i)*x_tp(4,i);
    x_t(:,i) = x_tp(:,i) + L*(V_expt(i) - V);
    x_t(1,i) = saturate(x_t(1,i),0,1);
    x_t(4,i) = growth(x_t(4,i), x_tp(4,i), max_growth);
    P_t = (eye(length(A)) - L*C)*P_tp;
    
    Voc = interp1(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), x_t(1,i), 'linear','extrap');
    Vb(i) = Voc - x_t(2,i) - x_t(3,i) - I_expt(i)*x_t(4,i);
    
    % Adapt
    if Adaptive == 1
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
end

prmse_SOC = calc_percent_rmse(SOC_CC,x_t(1,:));
prmse_V = calc_percent_rmse(V_expt,Vb);
disp(['RMS Error in SOC = ' num2str(prmse_SOC) '%']);
disp(['RMS Error in voltage = ' num2str(prmse_V) '%']);

cd(strcat(rootFolder,'\Results files'))
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
plot(t(1:length(Vb)),x_t(4,:))
xlabel('Time [s]'); ylabel('R0')

load handel
sound(y,Fs)


function [start_charge_idxs, end_charge_idxs] = find_charge_idxs(I,data_padding)
    start_charge_idxs = [];
    end_charge_idxs = [];
    for i = 2:1:length(I)
        delta_I = I(i) - I(i-1);
        if  delta_I < -1.5 && I(i-1) == 0
            start_charge_idxs = [start_charge_idxs,(i-1) - data_padding];
        elseif delta_I > 1.5 && I(i-1) < 0
            end_charge_idxs = [end_charge_idxs, i + data_padding];
        end
    end
end

function [start_discharge_idxs, end_discharge_idxs] = find_discharge_idxs(I,data_padding)
    start_discharge_idxs = [];
    end_discharge_idxs = [];
    for i = 2:1:length(I)
        delta_I = I(i) - I(i-1);
        if  delta_I > 1.5 && delta_I < 3 && I(i-1) == 0
            start_discharge_idxs = [start_discharge_idxs,(i-1) - data_padding];
        elseif delta_I < -1.5 && I(i-1) > 1.5 && I(i-1) < 3
            end_discharge_idxs = [end_discharge_idxs, i + data_padding];
        end
    end
end

function [SOC_map,R0,R1,C1,R2,C2] = estimate_parameters_graphically(start_idxs, end_idxs,t,V,I,SOC,data_padding)
    instant_delta_I_charging = I(start_idxs + data_padding + 1) - I(start_idxs + data_padding);
    instant_delta_V_charging = V(start_idxs + data_padding + 1) - V(start_idxs + data_padding);
    transient_delta_V = V(end_idxs - data_padding - 1) - V(start_idxs + data_padding + 1);
    transient_delta_t = t(end_idxs - data_padding - 1) - t(start_idxs + data_padding + 1);

    num_cycles = length(start_idxs);
    R0 = abs(instant_delta_V_charging)./abs(instant_delta_I_charging);
    R1 = zeros(num_cycles,1);
    C1 = zeros(num_cycles,1);
    R2 = zeros(num_cycles,1);
    C2 = zeros(num_cycles,1);
    SOC_map = zeros(num_cycles,1);
    %variables to aid calcs...
    V_oc = V(start_idxs + data_padding); %voltage at start before step in I
    V_tf = V(end_idxs - data_padding - 1); %voltage at end right before negative step in I
    V_t0_trans = V(start_idxs + data_padding + 1); %voltage at beginning of transient, after instantaneous response
    I0 = I(start_idxs + data_padding + 1); %step height
    order_magnitude = 10;
    for i = 1:1:num_cycles 
        syms R_1 C_1 R_2 C_2
        eqn1 = V_oc(i) - I0(i)*(R_1 + R_2 + R0(i)) == V_tf(i);
        eqn2 = order_magnitude*R_1*C_1 == R_2*C_2;
        if I0(i) < 0
            eqn3 = R_1*C_1 == transient_delta_t(i)/log((V_t0_trans(i) + transient_delta_V(i))/V_t0_trans(i));
        else
            eqn3 = R_1*C_1 == transient_delta_t(i)/log(V_t0_trans(i)/(V_t0_trans(i) + transient_delta_V(i)));
        end
        eqn4 = (1+rand)*R_1 == R_2;
        solution = solve([eqn1,eqn2,eqn3,eqn4],[R_1,C_1,R_2,C_2]);
        R1(i) = double(solution.R_1);
        C1(i) = double(solution.C_1);
        R2(i) = double(solution.R_2);
        C2(i) = double(solution.C_2);
        SOC_map(i) = SOC(start_idxs(i));
    end
    SOC_map = [1; SOC_map; 0];
    R0 = [R0(1); R0; R0(end)];
    R1 = [R1(1); R1; R1(end)];
    C1 = [C1(1); C1; C1(end)];
    R2 = [R2(1); R2; R2(end)];
    C2 = [C2(1); C2; C2(end)];
end
    
function [R0,R1,C1,R2,C2,percent_rms_error] = estimate_parameters_ga(start_idxs,end_idxs,t,V,I,SOC,SOC_map,OCV_map,ub,lb)    
    num_params = 5;
    num_cycles = length(start_idxs);    
    percent_rms_error = zeros(num_cycles,1);
    ECM_params_ga = zeros(num_cycles,num_params);
    for i = 1:1:num_cycles
        idxs = start_idxs(i):end_idxs(i); 
        [ga_param_est, ga_loss] = ga(@(ga_param_est)objective_function(ga_param_est,I(idxs),V(idxs),t(idxs),SOC(idxs),OCV_map,SOC_map),num_params,[],[],[],[],lb(i,:),ub(i,:));
        ECM_params_ga(i,:) = ga_param_est;
        percent_rms_error(i) = ga_loss;
    end
    percent_rms_error = [NaN; percent_rms_error; NaN];
    ECM_params_ga = [ECM_params_ga(1,:); ECM_params_ga; ECM_params_ga(end,:)];
    R0 = ECM_params_ga(:,1);
    R1 = ECM_params_ga(:,2);
    C1 = ECM_params_ga(:,3);
    R2 = ECM_params_ga(:,4);
    C2 = ECM_params_ga(:,5);
end

function obj_func_value = objective_function(params,I,V,t,SOC,OCV_map,SOC_map)
    N = length(V);
    V_sim = zeros(N,1);

    Voc = interp1(SOC_map,OCV_map,SOC);
    V_sim(1) = Voc(1);

    R0 = params(1);
    R1 = params(2);
    C1 = params(3);
    R2 = params(4);
    C2 = params(5);
    v1 = 0;
    v2 = 0;
    for i = 2:1:length(V_sim)
        dv1 = (I(i-1) - (v1/R1))/C1;
        dv2 = (I(i-1) - (v2/R2))/C2;
        v1 = v1 + (t(i) - t(i-1))*dv1;
        v2 = v2 + (t(i) - t(i-1))*dv2;
        V_sim(i) = Voc(i) - v1 - v2 - I(i)*R0;
    end
    obj_func_value = sqrt((1/N)*sum((V_sim - V).^2))*(100*N/sum(V));
end

function loss = objective_function_EKF_SOC_V1_V2_R0(ga_params,t,V,I,SOC_Coulomb_Counting,mu0,S0,R1,C1,R2,C2,Qn,p_dOCVdSOC_map,p_SOC_OCV_map,SOC_map,OCV_map)
    Q = diag(exp(ga_params(2:end)));
    R = exp(ga_params(1));
    [mu,S,Vb] = EKF_SOC_V1_V2_R0(t,V,I,mu0,S0,Q,R,R1,C1,R2,C2,Qn,p_dOCVdSOC_map,p_SOC_OCV_map,SOC_map,OCV_map);
    loss = calc_percent_rmse(SOC_Coulomb_Counting,mu(1,:));
    disp(loss);
end

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

function y = growth(x_current,x_prev,max_growth)
    if x_current < x_prev
        y = x_prev;
    elseif x_current > (1 + max_growth)*x_prev
        y = (1 + max_growth)*x_prev;
    else
        y = x_current;
    end
end

function prmse = calc_percent_rmse(x_actual,x_corrupted)
    N = length(x_actual);
    prmse = sqrt((1/N)*sum((reshape(x_corrupted,[],1) - reshape(x_actual,[],1)).^2))*(100*N/sum(x_actual));
end