clear all; close all; format compact; clc

%Non_adaptive
load('HPPCresult_N0');
prmse_V_HPPC_N0 = prmse_V;
prmse_SOC_HPPC_N0 = prmse_SOC;

load('HPPCresult_N75');
prmse_V_HPPC_N75 = prmse_V;
prmse_SOC_HPPC_N75 = prmse_SOC;

load('HPPCresult_N125');
prmse_V_HPPC_N125 = prmse_V;
prmse_SOC_HPPC_N125 = prmse_SOC;

load('HPPCresult_N200');
prmse_V_HPPC_N200 = prmse_V;
prmse_SOC_HPPC_N200 = prmse_SOC;

load('UDDSresult_N1');
prmse_V_UDDS_N1 = prmse_V;
prmse_SOC_UDDS_N1 = prmse_SOC;

load('UDDSresult_N201');
prmse_V_UDDS_N201 = prmse_V;
prmse_SOC_UDDS_N201 = prmse_SOC;

non_adaptive = [prmse_V_HPPC_N0;
                prmse_SOC_HPPC_N0;
                prmse_V_HPPC_N75;
                prmse_SOC_HPPC_N75;
                prmse_V_HPPC_N125;
                prmse_SOC_HPPC_N125;
                prmse_V_HPPC_N200;
                prmse_SOC_HPPC_N200;
                prmse_V_UDDS_N1;
                prmse_SOC_UDDS_N1;
                prmse_V_UDDS_N201;
                prmse_SOC_UDDS_N201];

%Adaptive
load('HPPCresult_N0_adaptive');
prmse_V_HPPC_N0_adaptive = prmse_V;
prmse_SOC_HPPC_N0_adaptive = prmse_SOC;

load('HPPCresult_N75_adaptive');
prmse_V_HPPC_N75_adaptive = prmse_V;
prmse_SOC_HPPC_N75_adaptive = prmse_SOC;

load('HPPCresult_N125_adaptive');
prmse_V_HPPC_N125_adaptive = prmse_V;
prmse_SOC_HPPC_N125_adaptive = prmse_SOC;

load('HPPCresult_N200_adaptive');
prmse_V_HPPC_N200_adaptive = prmse_V;
prmse_SOC_HPPC_N200_adaptive = prmse_SOC;

load('UDDSresult_N1_adaptive');
prmse_V_UDDS_N1_adaptive = prmse_V;
prmse_SOC_UDDS_N1_adaptive = prmse_SOC;

load('UDDSresult_N201_adaptive');
prmse_V_UDDS_N201_adaptive = prmse_V;
prmse_SOC_UDDS_N201_adaptive = prmse_SOC;

adaptive = [prmse_V_HPPC_N0_adaptive;
            prmse_SOC_HPPC_N0_adaptive;
            prmse_V_HPPC_N75_adaptive;
            prmse_SOC_HPPC_N75_adaptive;
            prmse_V_HPPC_N125_adaptive;
            prmse_SOC_HPPC_N125_adaptive;
            prmse_V_HPPC_N200_adaptive;
            prmse_SOC_HPPC_N200_adaptive;
            prmse_V_UDDS_N1_adaptive;
            prmse_SOC_UDDS_N1_adaptive;
            prmse_V_UDDS_N201_adaptive;
            prmse_SOC_UDDS_N201_adaptive];


T = table(non_adaptive,adaptive);
writetable(T,'prmse_table.xlsx');

load('HPPCresult_N0_HW4_EKF');
prmse_V_HPPC_N0_HW4_EKF = prmse_V;
prmse_SOC_HPPC_N0_HW4_EKF = prmse_SOC;

disp(['HPPC N0 HW4_EKF prmse Vb = ' num2str(prmse_V_HPPC_N0_HW4_EKF) '%']);
disp(['HPPC N0 HW4_EKF prmse SOC = ' num2str(prmse_SOC_HPPC_N0_HW4_EKF) '%']);

load('HPPCresult_N200_HW4_EKF');
prmse_V_HPPC_N200_HW4_EKF = prmse_V;
prmse_SOC_HPPC_N200_HW4_EKF = prmse_SOC;

disp(['HPPC N200 HW4_EKF prmse Vb = ' num2str(prmse_V_HPPC_N200_HW4_EKF) '%']);
disp(['HPPC N200 HW4_EKF prmse SOC = ' num2str(prmse_SOC_HPPC_N200_HW4_EKF) '%']);

