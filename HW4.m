%% Homework 4

% Authors: Miles, Jessica, Tom
% ENERGY 295
% 11/4/2021

close all;
clear;
clc;

%% EKF - without R0
load('HW4_batt_params.mat');
load('HW4_expt_data.mat');
t = 0:dt:((length(I_expt) - 1)*dt);
I = I_expt;
V = V_expt;
Qn = capacity;
SOC_map = Voc_vs_SOC(:,1);
OCV_map = Voc_vs_SOC(:,2);
SOC0 = SOC_init;
SOC_Coulomb_Counting = SOC0 - (cumtrapz(t, I)/3600)/Qn;

%Initial guess of mu and S
SOC0 = rand();
mu0 = [1;0;0];
S0 = diag([0.1,0,0]);
% S0 = diag([0.1,0.001,0.001]);


% % Finding Q from GA
% R = 8.432e-4;
% Q_diag = [1000*R,0.1*R,0.01*R];
% lb = log(0.5*[R Q_diag]);
% ub = log(1.5*[R Q_diag]);
% num_params = 4;
% max_time = 60*10;
% options = optimoptions('ga','PlotFcn',@gaplotbestf,'MaxTime',max_time);
% [ga_params, ga_loss] = ga(@(ga_params)objective_function_EKF_SOC_V1_V2(ga_params,t,V,I,SOC_Coulomb_Counting,mu0,S0,R0_chg,R0_dischg,R0_dischg_4A,R1_chg,R1_dischg,C1_chg,C1_dischg,R2_chg,R2_dischg,C2_chg,C2_dischg,soc_chg,soc_dischg,Qn,SOC_map,OCV_map),num_params,[],[],[],[],lb,ub,[],options);
% Q = diag(exp(ga_params(2:end)))
% R = exp(ga_params(1))


%Estimating Noise Matrices
% Q and R from lecture slides
R = 8.432e-4;
Q = diag([1000*R,0.1*R,0.01*R]);
[mu,S,Vb,K] = EKF_SOC_V1_V2(t,V,I,mu0,S0,Q,R,R0_chg,R0_dischg,R0_dischg_4A,R1_chg,R1_dischg,C1_chg,C1_dischg,R2_chg,R2_dischg,C2_chg,C2_dischg,soc_chg,soc_dischg,Qn,SOC_map,OCV_map);

%Plotting voltage prediciton based on mu_k|k
figure(); hold on;
plot(t,V,'DisplayName','Measured');
plot(t,Vb,'DisplayName','EKF Prediciton');
title('Voltage vs. Time');
xlabel('Time (sec)');
ylabel('Voltage (V)');
legend('Location','best');
percent_rmse_Vb = calc_percent_rmse(V,Vb);
disp(['RMS Error in Vb (EKF) = ' num2str(percent_rmse_Vb) '%']);

%Plotting SOC for Coulomb Counting and EKF
figure(); hold on;
plot(t,100*SOC_Coulomb_Counting,'DisplayName','Coulomb Counting');
plot(t,100*mu(1,:),'DisplayName','EKF Prediciton');
title('SOC vs. Time');
xlabel('Time (sec)');
ylabel('SOC (%)');
legend('Location','best');
percent_rmse_SOC = calc_percent_rmse(SOC_Coulomb_Counting,mu(1,:));
disp(['RMS Error in SOC (EKF vs. Coulomb Counting) = ' num2str(percent_rmse_SOC) '%']);

figure();
plot(t,K(1,:),t,K(2,:),t,K(3,:));
title('Kalman Gain vs. Time');
xlabel('Time (sec)'); ylabel('Gain');
legend('k1','k2','k3');


%% Vb Simulated without EKF
load('workspace.mat');
data = xlsread('INR21700_M50T_T23_HPPC_N0_W8.xlsx');
start_idx = 14476;
t = data(start_idx:end,2);
V = data(start_idx:end,3);
I = -data(start_idx:end,4);

SOC0 = 0 - (trapz(data(1:start_idx,2), -data(1:start_idx,4))/3600)/Qn;
SOC_Coulomb_Counting = SOC0 - (cumtrapz(t, I)/3600)/Qn;

[Vb,V1,V2,percent_rmse_Vb] = simulate(ECM_params_ga_ch,SOC_ch,ECM_params_ga_disch,SOC_disch,I,V,t,Qn,SOC0,OCV_map,SOC_map);

%Plotting voltage prediciton based on mu_k|k
figure(); hold on;
plot(t,V,'DisplayName','Measured');
plot(t,Vb,'DisplayName','Simulated Prediciton');
title('Voltage vs. Time');
xlabel('Time (sec)');
ylabel('Voltage (V)');
legend('Location','best');
percent_rmse_Vb = calc_percent_rmse(V,Vb);
disp(['RMS Error in Vb (EKF) = ' num2str(percent_rmse_Vb) '%']);

%% Functions

function [Vb,V1,V2,percent_rmse_Vb] = simulate(param_map_ch,SOC_param_map_ch,param_map_dis,SOC_param_map_dis,I,V,t,Qn,SOC0,OCV_map,SOC_OCV_map)

    N = length(V);
    Vb = zeros(N,1);
    Q = cumtrapz(t, I)/3600; %Ah
    SOC = SOC0 - Q/Qn;

    Voc = interp1(SOC_OCV_map,OCV_map,SOC);
    Vb(1) = Voc(1);

    R0_ch = interp1(SOC_param_map_ch,param_map_ch(:,1),SOC);
    R1_ch = interp1(SOC_param_map_ch,param_map_ch(:,2),SOC);
    C1_ch = interp1(SOC_param_map_ch,param_map_ch(:,3),SOC);
    R2_ch = interp1(SOC_param_map_ch,param_map_ch(:,4),SOC);
    C2_ch = interp1(SOC_param_map_ch,param_map_ch(:,5),SOC);

    R0_dis = interp1(SOC_param_map_dis,param_map_dis(:,1),SOC);
    R1_dis = interp1(SOC_param_map_dis,param_map_dis(:,2),SOC);
    C1_dis = interp1(SOC_param_map_dis,param_map_dis(:,3),SOC);
    R2_dis = interp1(SOC_param_map_dis,param_map_dis(:,4),SOC);
    C2_dis = interp1(SOC_param_map_dis,param_map_dis(:,5),SOC);

    r0 = 0;
    r1 = 0;
    c1 = 0;
    r2 = 0;
    c2 = 0;
    V1 = zeros(N,1);
    V2 = zeros(N,1);
    for i = 2:1:length(Vb)
        if I(i) < 0 
            r0 = R0_ch(i);
            r1 = R1_ch(i);
            c1 = C1_ch(i);
            r2 = R2_ch(i);
            c2 = C2_ch(i);
        else
            r0 = R0_dis(i);
            r1 = R1_dis(i);
            c1 = C1_dis(i);
            r2 = R2_dis(i);
            c2 = C2_dis(i);
        end

        dv1 = (I(i-1) - (V1(i-1)/r1))/c1;
        dv2 = (I(i-1) - (V2(i-1)/r2))/c2;
        V1(i) = V1(i-1) + (t(i) - t(i-1))*dv1;
        V2(i) = V2(i-1) + (t(i) - t(i-1))*dv2;
        Vb(i) = Voc(i) - V1(i) - V2(i) - I(i)*r0;
    end
    percent_rmse_Vb = sqrt((1/N)*sum((Vb - V).^2))*(100*N/sum(V));
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

function [mu,S,Vb,K] = EKF_SOC_V1_V2(t,V,I,mu0,S0,Q,R,R0_chg,R0_dischg,R0_dischg_4A,R1_chg,R1_dischg,C1_chg,C1_dischg,R2_chg,R2_dischg,C2_chg,C2_dischg,soc_chg,soc_dischg,Qn,SOC_map,OCV_map)
    N = length(t);
    num_states = 3;
    %initializing simulation parameters to be stored
    mu = zeros(num_states,N);%[SOC,R0,V1,V2]
    mu(:,1) = mu0;
    S = zeros(num_states,num_states,N);
    S(:,:,1) = S0;
    Vb = zeros(N,1);
    K = zeros(3,N);
    for k = 2:1:N
        %----predict step----
        % evaluate jacobian of A and B (evaluate at mu_k-1|k-1 --> A_k-1 and B_k-1)
        dT = t(k) - t(k-1);
        if I(k-1) < 0
            R1 = interp1(soc_chg,R1_chg,mu(1,k-1),'linear','extrap');
            C1 = interp1(soc_chg,C1_chg,mu(1,k-1),'linear','extrap');
            R2 = interp1(soc_chg,R2_chg,mu(1,k-1),'linear','extrap');
            C2 = interp1(soc_chg,C2_chg,mu(1,k-1),'linear','extrap');

            dR1_dSOC = find_d_dSOC(soc_chg,R1_chg,mu(1,k-1));
            dC1_dSOC = find_d_dSOC(soc_chg,C1_chg,mu(1,k-1));
            dR2_dSOC = find_d_dSOC(soc_chg,R2_chg,mu(1,k-1));
            dC2_dSOC = find_d_dSOC(soc_chg,C2_chg,mu(1,k-1));
        else
            R1 = interp1(soc_dischg,R1_dischg,mu(1,k-1),'linear','extrap');
            C1 = interp1(soc_dischg,C1_dischg,mu(1,k-1),'linear','extrap');
            R2 = interp1(soc_dischg,R2_dischg,mu(1,k-1),'linear','extrap');
            C2 = interp1(soc_dischg,C2_dischg,mu(1,k-1),'linear','extrap');

            dR1_dSOC = find_d_dSOC(soc_dischg,R1_dischg,mu(1,k-1));
            dC1_dSOC = find_d_dSOC(soc_dischg,C1_dischg,mu(1,k-1));
            dR2_dSOC = find_d_dSOC(soc_dischg,R2_dischg,mu(1,k-1));
            dC2_dSOC = find_d_dSOC(soc_dischg,C2_dischg,mu(1,k-1));
        end

        A = [0, 0, 0;
             (C1*dR1_dSOC + R1*dC1_dSOC)/((R1*C1)^2), -1/(R1*C1), 0;
             (C2*dR2_dSOC + R2*dC2_dSOC)/((R2*C2)^2), 0, -1/(R2*C2)];
%         A = exp(A*dT);
        % mean: mu_k|k-1
        mu_predict = mu(:,k-1) + dT*[-I(k-1)*(1/3600)/Qn; ...
                                    (-mu(2,k-1)/(R1*C1)) + (I(k-1)/C1); ...
                                    (-mu(3,k-1)/(R2*C2)) + (I(k-1)/C2)];
        %constraining SOC
        mu_predict(1) = saturate(mu_predict(1),0,1);
        %covariance: S_k|k-1
        S_predict = A*S(:,:,k-1)*A' + Q; 
    
    
        %----update step----
        %evaluate jacobian of C (Evaluate Ck at mu_k|k-1)
        dVoc_dSOC = find_d_dSOC(SOC_map,OCV_map,mu_predict(1));
        if I(k) < 0
            R0 = interp1(soc_chg,R0_chg,mu_predict(1),'linear','extrap');
            dR0_dSOC = find_d_dSOC(soc_chg,R0_chg,mu_predict(1));
        else
            if I(k) >= 4
                R0 = interp1(soc_dischg,R0_dischg_4A,mu_predict(1),'linear','extrap');
                dR0_dSOC = find_d_dSOC(soc_dischg,R0_dischg_4A,mu_predict(1));
            else
                R0 = interp1(soc_dischg,R0_dischg,mu_predict(1),'linear','extrap');
                dR0_dSOC = find_d_dSOC(soc_dischg,R0_dischg,mu_predict(1));
            end
        end
        C = [(dVoc_dSOC - dR0_dSOC*I(k)), -1, -1];
        %Kalman Gain
        K(:,k) = S_predict*C'*inv(C*S_predict*C' + R); 
        % mean: mu_k|k
        Voc = interp1(SOC_map,OCV_map,mu_predict(1),'linear','extrap');
        mu(:,k) = mu_predict + K(:,k)*(V(k) - (Voc - mu_predict(2) - mu_predict(3) - I(k)*R0)); %Evaluate at mu_k|k-1
        %constraining SOC
        mu(1,k) = saturate(mu(1,k),0,1);
        %covariance: S_k|k
        S(:,:,k) = S_predict - K(:,k)*C*S_predict; %Evaluate at S_k|k-1,Ck
        %Finding 
        Voc = interp1(SOC_map,OCV_map,mu(1,k),'linear','extrap');
        if I(k) < 0
            R0 = interp1(soc_chg,R0_chg,mu(1,k),'linear','extrap');
        else
            if I(k) >= 4
                R0 = interp1(soc_dischg,R0_dischg_4A,mu(1,k),'linear','extrap');
            else
                R0 = interp1(soc_dischg,R0_dischg,mu(1,k),'linear','extrap');
            end
        end
        Vb(k) = Voc - mu(2,k) - mu(3,k) - I(k)*R0; 
    end
end

function loss = objective_function_EKF_SOC_V1_V2(ga_params,t,V,I,SOC_Coulomb_Counting,mu0,S0,R0_chg,R0_dischg,R0_dischg_4A,R1_chg,R1_dischg,C1_chg,C1_dischg,R2_chg,R2_dischg,C2_chg,C2_dischg,soc_chg,soc_dischg,Qn,SOC_map,OCV_map)
    Q = diag(exp(ga_params(2:end)));
    R = exp(ga_params(1));
    [mu,S,Vb,K] = EKF_SOC_V1_V2(t,V,I,mu0,S0,Q,R,R0_chg,R0_dischg,R0_dischg_4A,R1_chg,R1_dischg,C1_chg,C1_dischg,R2_chg,R2_dischg,C2_chg,C2_dischg,soc_chg,soc_dischg,Qn,SOC_map,OCV_map);
    loss = calc_percent_rmse(SOC_Coulomb_Counting,mu(1,:));
    disp(loss);
end

function prmse = calc_percent_rmse(x_actual,x_corrupted)
    N = length(x_actual);
    prmse = sqrt((1/N)*sum((reshape(x_corrupted,[],1) - reshape(x_actual,[],1)).^2))*(100*N/sum(x_actual));
end

function dOut_dIn = find_d_dSOC(input_map,output_map,query_point)
    delta = 0.01;
    output_high = interp1(input_map,output_map,query_point + delta,'linear','extrap');
    output_low = interp1(input_map,output_map,query_point - delta,'linear','extrap');
    dOut_dIn = (output_high - output_low)/(2*delta); 
end
