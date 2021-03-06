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
R0 = mean([R0_chg,R0_dischg]);
R1 = mean([R1_chg,R1_dischg]);
C1 = mean([C1_chg,C1_dischg]);
R2 = mean([R2_chg,R2_dischg]);
C2 = mean([C2_chg,C2_dischg]);

SOC0 = SOC_init;
SOC_Coulomb_Counting = SOC0 - (cumtrapz(t, I)/3600)/Qn;


ECM_params_ga_ch = [R0_chg;R1_chg;C1_chg;R2_chg;C2_chg]';
SOC_ch = soc_chg;
ECM_params_ga_disch = [R0_dischg;R1_dischg;C1_dischg;R2_dischg;C2_dischg]';
SOC_disch = soc_dischg;
[V,V1,V2,percent_rmse_Vb] = simulate(ECM_params_ga_ch,SOC_ch,ECM_params_ga_disch,SOC_disch,I,V,t,Qn,SOC0,OCV_map,SOC_map);


%Initial guess of mu and S
SOC0 = rand();
mu0 = [SOC0;0;0]; %Assume starting at rest --> V1_0 = 0, V2_0 = 0
S0 = diag([0.1,0.001,0.001]);

% % Finding Q from GA
% R = 8.432e-4;
% Q_diag = [1000*R,0.1*R,0.01*R];
% lb = log(0.5*[R Q_diag]);
% ub = log(1.5*[R Q_diag]);
% num_params = 4;
% max_time = 60*10;
% options = optimoptions('ga','PlotFcn',@gaplotbestf,'MaxTime',max_time);
% [ga_params, ga_loss] = ga(@(ga_params)objective_function_EKF_SOC_V1_V2(ga_params,t,V,I,SOC_Coulomb_Counting,mu0,S0,R0,R1,C1,R2,C2,Qn,SOC_map,OCV_map),num_params,[],[],[],[],lb,ub,[],options);
% Q = diag(exp(ga_params(2:end)))
% R = exp(ga_params(1))


%Estimating Noise Matrices
% Q and R from lecture slides
R = 8.432e-4;
Q = diag([1000*R,0.1*R,0.01*R]);
% Q = zeros(3,3);

[mu,S,Vb,K] = EKF_SOC_V1_V2(t,V,I,mu0,S0,Q,R,R0,R1,C1,R2,C2,Qn,SOC_map,OCV_map);

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

%% EKF - with R0
load('HW4_batt_params.mat');
load('HW4_expt_data.mat');
t = 0:dt:((length(I_expt) - 1)*dt);
I = I_expt;
V = V_expt;
Qn = capacity;
SOC_map = Voc_vs_SOC(:,1);
OCV_map = Voc_vs_SOC(:,2);
R0 = mean([R0_chg,R0_dischg]);
R1 = mean([R1_chg,R1_dischg]);
C1 = mean([C1_chg,C1_dischg]);
R2 = mean([R2_chg,R2_dischg]);
C2 = mean([C2_chg,C2_dischg]);

SOC0 = SOC_init;
SOC_Coulomb_Counting = SOC0 - (cumtrapz(t, I)/3600)/Qn;


% ECM_params_ga_ch = [R0_chg;R1_chg;C1_chg;R2_chg;C2_chg]';
% SOC_ch = soc_chg;
% ECM_params_ga_disch = [R0_dischg;R1_dischg;C1_dischg;R2_dischg;C2_dischg]';
% SOC_disch = soc_dischg;
% [V,V1,V2,percent_rmse_Vb] = simulate(ECM_params_ga_ch,SOC_ch,ECM_params_ga_disch,SOC_disch,I,V,t,Qn,SOC0,OCV_map,SOC_map);

SOC0 = 0 - (trapz(data(1:start_idx,2), -data(1:start_idx,4))/3600)/Qn;
SOC_Coulomb_Counting = SOC0 - (cumtrapz(t, I)/3600)/Qn;

R0 = mean([ECM_params_ga_disch(:,1); ECM_params_ga_ch(:,1)]);
R1 = mean([ECM_params_ga_disch(:,2); ECM_params_ga_ch(:,2)]);
C1 = mean([ECM_params_ga_disch(:,3); ECM_params_ga_ch(:,3)]);
R2 = mean([ECM_params_ga_disch(:,4); ECM_params_ga_ch(:,4)]);
C2 = mean([ECM_params_ga_disch(:,5); ECM_params_ga_ch(:,5)]);

%Initial estimates 
SOC0 = rand();
mu0 = [SOC0;0;0;R0]; 
S0 = diag([0.1,0.001,0.001,0.01]);
% S0 = diag([(SOC0 - SOC_Coulomb_Counting(1))^2,0,0,0]);

% % Finding Q from GA
% num_params = 5;
% lb = -7*ones(num_params,1);
% ub = 0*ones(num_params,1);
% max_time = 60*10;
% options = optimoptions('ga','PlotFcn',@gaplotbestf,'MaxTime',max_time);
% [ga_params, ga_loss] = ga(@(ga_params)objective_function_EKF_SOC_V1_V2_R0(ga_params,t,V,I,SOC_Coulomb_Counting,mu0,S0,R1,C1,R2,C2,Qn,SOC_map,OCV_map),num_params,[],[],[],[],lb,ub,[],options);
% Q = diag(exp(ga_params(2:end)))
% R = exp(ga_params(1))

%%
% %From lecture slides
% R = 1e-4;
% % Q = diag([100*R,0.1*R,0.01*R,0.1*R]);

% %From GA with Q
% R = 1e-4;
% Q = [1.9474        0         0         0;
%          0    0.0733         0         0;
%          0         0    0.0171         0;
%          0         0         0    0.0019]; %<--these values found from GA above (see GA_convergence.fig)

% From GA with Q and R as a params
R = 131.8108;
Q =  [493.7918         0         0         0;
             0    0.0031         0         0;
             0         0    0.0018         0;
             0         0         0    0.6748];

[mu,S,Vb] = EKF_SOC_V1_V2_R0(t,V,I,mu0,S0,Q,R,R1,C1,R2,C2,Qn,SOC_map,OCV_map);

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

%Plotting R0 vs. time
figure(); hold on;
plot(t,mu(4,:));
title('R0 vs. Time');
xlabel('Time (sec)');
ylabel('R0 (Ohms)');

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
    
function [R0,R1,C1,R2,C2,percent_rms_error] = estimate_parameters_ga(start_idxs, end_idxs,t,V,I,SOC,SOC_map,OCV_map,ub,lb)    
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

function [mu,S,Vb,K] = EKF_SOC_V1_V2(t,V,I,mu0,S0,Q,R,R0,R1,C1,R2,C2,Qn,SOC_map,OCV_map)
    tau1 = R1*C1;
    tau2 = R2*C2;
    N = length(t);
    num_states = 3;
    %initializing mean and covariance
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
        Akm1 = [1,0,0;
                0,exp(-dT/tau1),0;
                0,0,exp(-dT/tau2)];
        Bkm1 = [-(dT/3600)/Qn; R1*(1 - exp(-dT/tau1)); R2*(1 - exp(-dT/tau2))];%Notation: km1 = k-1 
        % mean: mu_k|k-1
        mu_predict = Akm1*mu(:,k-1) + Bkm1*I(k-1); 
        %constraining SOC
        mu_predict(1) = saturate(mu_predict(1),0,1);
        %covariance: S_k|k-1
        S_predict = Akm1*S(:,:,k-1)*Akm1' + Q; 
    
    
        %----update step----
        %evaluate jacobian of C (Evaluate Ck at mu_k|k-1)
        dVoc_dSOC = find_dVoc_dSOC(SOC_map,OCV_map,mu_predict(1));
        Ck = [dVoc_dSOC, -1, -1];
        %Kalman Gain
        K(:,k) = S_predict*Ck'*inv(Ck*S_predict*Ck' + R); 
        % mean: mu_k|k
        Voc = interp1(SOC_map,OCV_map,mu_predict(1),'linear','extrap');
        mu(:,k) = mu_predict + K(:,k)*(V(k) - (Voc - mu_predict(2) - mu_predict(3) - I(k)*R0)); %Evaluate at mu_k|k-1
        %constraining SOC
        mu(1,k) = saturate(mu(1,k),0,1);
        %covariance: S_k|k
        S(:,:,k) = S_predict - K(:,k)*Ck*S_predict; %Evaluate at S_k|k-1,Ck
        Voc = interp1(SOC_map,OCV_map,mu(1,k),'linear','extrap');
        Vb(k) = Voc - mu(2,k) - mu(3,k) - I(k)*R0; %polyval(p_SOC_OCV_map,mu(1,k)) % yk = g(mu_k|k)
    end
end

function [mu,S,Vb] = EKF_SOC_V1_V2_R0(t,V,I,mu0,S0,Q,R,R1,C1,R2,C2,Qn,SOC_map,OCV_map)
    tau1 = R1*C1;
    tau2 = R2*C2;
    N = length(t);
    max_growth = 0.05;
    num_states = 4;
    %Predicted Measurement
    Vb = zeros(N,1);
    %initializing mean and covariance
    mu = zeros(num_states,N);%[SOC,R0,V1,V2]
    mu(:,1) = mu0;
    S = zeros(num_states,num_states,N);
    S(:,:,1) = S0;
    for k = 2:1:N  
        %----predict step----
        % evaluate jacobian of A and B (evaluate at mu_k-1|k-1 --> A_k-1 and B_k-1)
        dT = t(k) - t(k-1);
        Akm1 = [1,0,0,0;
                0,exp(-dT/tau1),0,0;
                0,0,exp(-dT/tau2),0;
                0,0,0,1]; % Notation: km1 = k-1 
        Bkm1 = [-(dT/3600)/Qn; R1*(1 - exp(-dT/tau1)); R2*(1 - exp(-dT/tau2));0];% Notation: km1 = k-1
        % mean: mu_k|k-1
        mu_predict = Akm1*mu(:,k-1) + Bkm1*I(k-1); 
        %constraining SOC and R0
        mu_predict(1) = saturate(mu_predict(1),0,1);
        mu_predict(4) = growth(mu_predict(4),mu(4,k-1),max_growth);
        %covariance: S_k|k-1
        S_predict = Akm1*S(:,:,k-1)*Akm1' + Q; 
    
    
        %----update step----
        %evaluate jacobian of C (Evaluate Ck at mu_k|k-1)
        dVoc_dSOC = find_dVoc_dSOC(SOC_map,OCV_map,mu_predict(1));
        Ck = [dVoc_dSOC, -1, -1, -I(k)];
        %Kalman Gain
        K = S_predict*Ck'*inv(Ck*S_predict*Ck' + R); 
        % mean: mu_k|k
        Voc = interp1(SOC_map,OCV_map,mu_predict(1),'linear','extrap');
        mu(:,k) = mu_predict + K*(V(k) - (Voc - mu_predict(2) - mu_predict(3) - I(k)*mu_predict(4))); %Evaluate at mu_k|k-1
        %constraining SOC and R0
        mu(1,k) = saturate(mu(1,k),0,1);
        mu(4,k) = growth(mu(4,k),mu(4,k-1),max_growth);
        %covariance: S_k|k
        S(:,:,k) = S_predict - K*Ck*S_predict; %Evaluate at S_k|k-1,Ck
        Voc = interp1(SOC_map,OCV_map,mu(1,k),'linear','extrap');
        Vb(k) = Voc - mu(2,k) - mu(3,k) - I(k)*mu(4,k); % yk = g(mu_k|k) 
    end
end

function loss = objective_function_EKF_SOC_V1_V2_R0(ga_params,t,V,I,SOC_Coulomb_Counting,mu0,S0,R1,C1,R2,C2,Qn,SOC_map,OCV_map)
    Q = diag(exp(ga_params(2:end)));
    R = exp(ga_params(1));
    [mu,S,Vb] = EKF_SOC_V1_V2_R0(t,V,I,mu0,S0,Q,R,R1,C1,R2,C2,Qn,SOC_map,OCV_map);
    loss = calc_percent_rmse(SOC_Coulomb_Counting,mu(1,:));
    disp(loss);
end

function loss = objective_function_EKF_SOC_V1_V2(ga_params,t,V,I,SOC_Coulomb_Counting,mu0,S0,R0,R1,C1,R2,C2,Qn,SOC_map,OCV_map)
    Q = diag(exp(ga_params(2:end)));
    R = exp(ga_params(1));
    [mu,S,Vb,K] = EKF_SOC_V1_V2(t,V,I,mu0,S0,Q,R,R0,R1,C1,R2,C2,Qn,SOC_map,OCV_map);
    loss = calc_percent_rmse(SOC_Coulomb_Counting,mu(1,:));
    disp(loss);
end

function prmse = calc_percent_rmse(x_actual,x_corrupted)
    N = length(x_actual);
    prmse = sqrt((1/N)*sum((reshape(x_corrupted,[],1) - reshape(x_actual,[],1)).^2))*(100*N/sum(x_actual));
end

function dVoc_dSOC = find_dVoc_dSOC(SOC_map,OCV_map,SOC)
    delta = 0.01;
    Voc_high = interp1(SOC_map,OCV_map,SOC + delta,'linear','extrap');
    Voc_low = interp1(SOC_map,OCV_map,SOC - delta,'linear','extrap');
    dVoc_dSOC = (Voc_high - Voc_low)/(2*delta); 
end
