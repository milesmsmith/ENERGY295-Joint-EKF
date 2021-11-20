function Q_estim = adaptive(Vexp,Vb,mu,mu_predict,C)

% if length(Vb)<10
%     Q_estim = adaptive(Vexp(1:k),Vb,mu,mu_predict,C);
% else
%     Q_estim = adaptive(Vexp(k-N:k),Vb(k-N:k),mu,mu_predict,C);
% end
    
    d = Vexp - Vb;
    D = 1/N * sum(d*transpose(d))
    
    delta_mu = mu - mu_predict;
    L = d/delta_mu;
    
    Q_estim = L*D*transpose(L);
end

