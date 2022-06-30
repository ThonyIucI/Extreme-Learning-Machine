function [Performance,Error,FIT,B,W,b]=ELM(X,Y,Xtest,Ytest,n_neuronas,i)                 
% X: Inputs de entrenamiento [Ni,1]
% Y: Targets de entrenamiento [Ni,1]
% Xtest: Inputs de testeo [Nt,1]
% Ytest: Targets de testeo [Nt,1]
% Ni: número de muestras de identificación
% Nt: número de muestras para testeo
% n_neuronas: Número de neuronas usado
% Se devuelve el performance de identificación, Error de testeo, los pesos
% de salida (B) y los pesos de entrada (W).

%% Identificación
W=rand(n_neuronas,size(X,2))*2-1; %Pesos de entrada (W)
b=(rand(n_neuronas,size(X,2)));%Bias de la capa oculta
H = 1 ./ (1 + exp(-(W*X'+b))); %Calculo de la capa oculta (Función Sigmoide)
B=pinv(H') * Y ; %Pesos de salida (Beta) %Pseudoinversa de Moore-Penrose
Yout_I=(H' * B)' ; %Salida del modelo
Performance=sqrt(mse(Y'-Yout_I));%Cálculo de performance

%% Testeo
Z = W * Xtest' + b;
Ho = 1 ./ (1 + exp(-Z));
Yout_T = Ho' * B ;
Error = mse(Ytest - Yout_T);
FIT = 100 * (1-norm(Yout_T-Ytest)/norm(Yout_T-mean(Yout_T)));
end
