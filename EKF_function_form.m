function [x_t,Vb,L] = EKF(dt,V_expt,I_expt,x_t,P_t,Q,R,R0_chg,R0_dischg,R0_dischg_4A,...
        R1_chg,R1_dischg,C1_chg,C1_dischg,R2_chg,R2_dischg,C2_chg,C2_dischg,...
        soc_chg,soc_dischg,Qn,Voc_vs_SOC,deltaSOC,Adaptive,W)
    N = length(V_expt);
    for i = 2:N
        if I_expt(i-1) < 0
%             R0 = interp1(soc_chg, R0_chg, x_t(1,i-1), 'linear','extrap');
            R1 = interp1(soc_chg, R1_chg, x_t(1,i-1), 'linear','extrap');
            R2 = interp1(soc_chg, R2_chg, x_t(1,i-1), 'linear','extrap');
            C1 = interp1(soc_chg, C1_chg, x_t(1,i-1), 'linear','extrap');
            C2 = interp1(soc_chg, C2_chg, x_t(1,i-1), 'linear','extrap');
%             dR0 = derivative(soc_chg, R0_chg, x_t(1,i-1),deltaSOC);
            dR1 = derivative(soc_chg, R1_chg, x_t(1,i-1),deltaSOC);
            dR2 = derivative(soc_chg, R2_chg, x_t(1,i-1),deltaSOC);
            dC1 = derivative(soc_chg, C1_chg, x_t(1,i-1),deltaSOC);
            dC2 = derivative(soc_chg, C2_chg, x_t(1,i-1),deltaSOC);
        else
%             if I_expt(i-1)>=4
%                 R0 = interp1(soc_chg, R0_dischg_4A, x_t(1,i-1), 'linear','extrap');
%                 dR0 = derivative(soc_dischg, R0_dischg_4A, x_t(1,i-1),deltaSOC);
%             else
%                 R0 = interp1(soc_chg, R0_dischg, x_t(1,i-1), 'linear','extrap');
%                 dR0 = derivative(soc_dischg, R0_dischg, x_t(1,i-1),deltaSOC);
%             end
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
        A = [0, 0, 0, 0; ...
             ( x_t(2,i-1)*(R1*dC1 + C1*dR1)/(R1*C1)^2 - I_expt(i-1)*dC1/C1^2 ), ( -1/(R1*C1) ), 0,              0;
             ( x_t(3,i-1)*(R2*dC2 + C2*dR2)/(R2*C2)^2 - I_expt(i-1)*dC2/C2^2 ), 0,              ( -1/(R2*C2) ), 0;
             0,                                                                 0,              0,              1];
        B = [-1/(3600*Qn); 1/C1; 1/C2; 0];
        C = [dV_dSOCV, -1, -1, -I_expt(i-1)];
        
        % Predict
        x_tp(:,i) = x_t(:,i-1) + dt*[-I_expt(i-1)/(3600*Qn);...
                                    -x_t(2,i-1)/(R1*C1) + I_expt(i-1)/C1;...
                                    -x_t(3,i-1)/(R2*C2) + I_expt(i-1)/C2;
                                    x_t(4,i-1)];
        x_tp(1,i) = saturate(x_tp(1,i),0,1);
        P_tp = A*P_t*A' + Q;
        
        % Correct
        L(:,i) = P_tp * C' * inv(C*P_tp*C' + R);
        Voc = interp1(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), x_tp(1,i),'linear','extrap');
        V = Voc - x_tp(2,i) - x_tp(3,i) - I_expt(i)*x_tp(4,i);
        x_t(:,i) = x_tp(:,i) + L(:,i)*(V_expt(i) - V);
        x_t(1,i) = saturate(x_t(1,i),0,1);
        P_t = (eye(length(A)) - L(:,i)*C)*P_tp;
        
        Voc = interp1(Voc_vs_SOC(:,1), Voc_vs_SOC(:,2), x_t(1,i), 'linear','extrap');
        Vb(i) = Voc - x_t(2,i) - x_t(3,i) - I_expt(i)*x_t(4,i);

        if adaptive
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
end