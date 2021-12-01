clear all; close all; format compact; clc

figure(); set(gcf,'color','w')

subplot(2,3,1); hold on
load('HPPCresult_N0')
plot(t,SOC_CC*100,'DisplayName','Experimental')
plot(t,x_t(1,:)*100,'DisplayName','EKF')
title('HPPC N = 0'); xlabel('Time [s]'); ylabel('SOC [%]'); legend('show')

subplot(2,3,2); hold on
load('HPPCresult_N75')
plot(t,SOC_CC*100,'DisplayName','Experimental')
plot(t,x_t(1,:)*100,'DisplayName','EKF')
title('HPPC N = 75'); xlabel('Time [s]'); ylabel('SOC [%]'); legend('show')

subplot(2,3,3); hold on
load('HPPCresult_N125')
plot(t,SOC_CC*100,'DisplayName','Experimental')
plot(t,x_t(1,:)*100,'DisplayName','EKF')
title('HPPC N = 125'); xlabel('Time [s]'); ylabel('SOC [%]'); legend('show')

subplot(2,3,4); hold on
load('HPPCresult_N200')
plot(t,SOC_CC*100,'DisplayName','Experimental')
plot(t,x_t(1,:)*100,'DisplayName','EKF')
title('HPPC N = 200'); xlabel('Time [s]'); ylabel('SOC [%]'); legend('show')

subplot(2,3,5); hold on
load('UDDSresult_N201')
plot(t,SOC_CC*100,'DisplayName','Experimental')
plot(t,x_t(1,:)*100,'DisplayName','EKF')
title('UDDS N = 201'); xlabel('Time [s]'); ylabel('SOC [%]'); legend('show')



figure(); set(gcf,'color','w'); hold on;

subplot(2,1,1); hold on
load('HPPCresult_N0')
plot(t,x_t(4,:),'DisplayName','N=0')

load('HPPCresult_N75')
plot(t,x_t(4,:),'DisplayName','N=75')

load('HPPCresult_N125')
plot(t,x_t(4,:),'DisplayName','N=125')

load('HPPCresult_N200')
plot(t,x_t(4,:),'DisplayName','N=200')

title('HPPC'); xlabel('Time [s]'); ylabel('R_0'); legend('show')

subplot(2,1,2); hold on
load('UDDSresult_N201')
plot(t,x_t(4,:))
title('UDDS'); xlabel('Time [s]'); ylabel('R_0');