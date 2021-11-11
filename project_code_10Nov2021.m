clear all; close all; format compact; clc;
univ_params.SOC0 = 1;
rootFolder = cd;

%% Open circuit voltage data
cd(strcat(rootFolder,'\Project_2_Data'))

data = readmatrix('INR21700_M50T_T23_OCV_W8.xlsx');
OCV_data.t = data(:,1);
OCV_data.V = data(:,2);
OCV_data.I = -data(:,3);

OCV_data.Q   = cumtrapz(OCV_data.t,OCV_data.I)/3600;
OCV_data.Qn  = OCV_data.Q(end);
OCV_data.SOC = ( univ_params.SOC0 - OCV_data.Q/OCV_data.Qn )*100;

figure; set(gcf,'color','w')
plot(OCV_data.SOC, OCV_data.V, 'linewidth',2)
xlabel('SOC [%]'); ylabel('OCV [V]')

%% Capacity values
data_capacity  = readmatrix('Capacity_Values.xlsx');
univ_params.N  = data_capacity(:,1);
univ_params.Qn = data_capacity(:,2);

%% HPPC data
cd(strcat(rootFolder,'\Project_2_Data','\HPPC'))

fileList = dir('**/*.xlsx');
for k = 1:length(fileList)
    a = getfield(fileList(k),'name');
    data = readmatrix(a);
    HPPC_data(k).t = data(19300:52346,2);
    HPPC_data(k).V = data(19300:52346,3);
    HPPC_data(k).I = -data(19300:52346,4);
        a = erase(a,'INR21700_M50T_T23_HPPC_N');
        a = str2num(erase(a,'_W8.xlsx'));
    HPPC_data(k).N  = a;
    HPPC_data(k).Qn = univ_params.Qn( find(a==univ_params.N) );
%     figure; plot(HPPC_data(k).t,HPPC_data(k).I)
    
    HPPC_data(k).Q = cumtrapz(HPPC_data(k).t,HPPC_data(k).I)/3600;
    HPPC_data(k).SOC = ( univ_params.SOC0 - HPPC_data(k).Q/HPPC_data(k).Qn )*100;
%     figure; plot(HPPC_data(k).SOC,HPPC_data(k).V)
end

%% HPPC data: parsing
cd(rootFolder)

for k = 1:length(HPPC_data)
    index = find( ischange(HPPC_data(k).I) ==1);
    t = HPPC_data(k).t;
    V = HPPC_data(k).V;
    I = HPPC_data(k).I;
%     figure; hold on; plot(t,I)

    j=1;
    for i=1:8
        R0(i) = abs( V(index(j)) - V(index(j)-1) ) / I(index(j));
        R1(i) = V(index(j)-1) - V(index(j+1)-1) - R0(i);
        C1(i) = (  t(index(j+1)-1) - t(index(j)-1)  )/(4*R1(i));
        R2(i) = 2*R1(i);
        C2(i) = 5*C1(i);
        tau1(i) = R1(i)*C1(i);
        tau2(i) = R2(i)*C2(i);

        i_endD(i) = ( find(t==t(index(j+2)-1)) + find(t== t(index(j+1))) )/2;
        i_startC(i) = i_endD(i);
        di(i) = i_endD(i) - find(t== t(index(j+1)));
        i_startD(i) = find(t==t(index(j)-1)) - di(i);
        i_endC(i) = find(t== t(index(j+3))) + di(i);

%         plot(t(i_startD),I(i_startD),'b.','MarkerSize',12)
%         plot(t(i_endD),I(i_endD),'b.','MarkerSize',12)
%         plot(t(i_startC),I(i_startC),'b.','MarkerSize',12)
%         plot(t(i_endC),I(i_endC),'b.','MarkerSize',12)
%         plot(t(index(j)-1),I(index(j)-1),'k.','MarkerSize',12)
%         plot(t(index(j)),I(index(j)),'k.','MarkerSize',12)
%         plot(t(index(j+1)-1),I(index(j+1)-1),'k.','MarkerSize',12)
%         plot(t(index(j+2)-1),I(index(j+2)-1),'k.','MarkerSize',12)
%         plot(t(index(j+2)),I(index(j+2)),'k.','MarkerSize',12)
%         plot(t(index(j+3)-1),I(index(j+3)-1),'k.','MarkerSize',12)

        j=j+6;
    end

    HPPC_data(k).R0 = R0;
    HPPC_data(k).R1 = R1;
    HPPC_data(k).C1 = C1;
    HPPC_data(k).R2 = R2;
    HPPC_data(k).C2 = C2;
    HPPC_data(k).tau1 = R1.*C1;
    HPPC_data(k).tau2 = R2.*C2;
    HPPC_data(k).i_endD   = i_endD;
    HPPC_data(k).i_startC = i_startC;
    HPPC_data(k).i_startD = i_startD;
    HPPC_data(k).i_endC   = i_endC;
end
clear t V I

%% UDDS data
cd(strcat(rootFolder,'\Project_2_Data','\UDDS'))
fileList = dir('**/*.xlsx');
for i = 1:length(fileList)
    a = getfield(fileList(i),'name');
    data = readmatrix(a);
    UDDS_data(i).t  = data(:,2);
    UDDS_data(i).V  = data(:,3);
    UDDS_data(i).I  = -data(:,4);
        a = erase(a,'INR21700_M50T_T23_UDDS_W8_N');
        a = str2num(erase(a,'.xlsx'));
    UDDS_data(i).N  = a;
end

%% Function definitions
function Vsim = simCalc(HPPC_data,OCV_data,univ_params,k)
    % Use index k to indicate the HPPC dataset of interet
    
    N = length(HPPC_data(k).V)
    V1(1) = 0;
    V2(1) = 0;
    for i = 2:N
        dt     = HPPC_data(k).t(i) - HPPC_data(k).t(i-1)
        dV1_dt = HPPC_data(k).I(i-1)/HPPC_data(k).C1...
                    - V1(i-1)/HPPC_data(k).tau1;
        dV2_dt = HPPC_data(k).I(i-1)/HPPC_data(k).C2...
                    - V2(i-1)/HPPC_data(k).tau2;
        V1(i)  = V1(i-1) + dV1_dt*dt;
        V2(i)  = V2(i-1) + dV2_dt*dt;
    end
    Voc  = interp1(OCV_data.SOC,OCV_data.V,HPPC_data.SOC);
    Vsim = Voc - HPPC_data(k).R0*HPPC_data(k).I - V1' - V2';
    
    RMS = sqrt( 1/N *sum((HPPC_data(k).V-Vsim).^2) )...
                * 100*N/sum(HPPC_data(k).V);
end