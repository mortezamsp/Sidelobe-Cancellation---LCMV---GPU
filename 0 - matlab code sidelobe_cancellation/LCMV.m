

clear all; 
close all;
clc
%SNR=10;
%diBi=1/(2*SNR)^0.5;

format long;%%���ü��㾫��
array=25;
Num_signal=10;
b=zeros(array,1);
B=zeros(array,Num_signal);
doa=zeros(Num_signal,1);
AA=zeros(1,array);
l=0.06;%����  f=5 G;
d=0.5*l;%��Ԫ���
e=[1 0 0 0 0 0 0 0 0 0 ]';
Len_train=100;
Len=100000;

x_train=zeros(Num_signal,Len_train);%signal for training
N_train=zeros(array,Len_train);%noise for training

xxx=zeros(Num_signal,Len);%signal
Noise1=zeros(array,Len);%noise

Plot_SNR=-6:2:18; %%%�����õ������
Plot_Pe=[];       %%%�˾������������洢�����������

W=zeros(array,1);

x1=-0.6*pi/2;x2=-pi/3;x3=-pi/4;x4=-pi/6;x5=0.9*pi;x6=pi/6;x7=pi/4;x8=pi/3;x9=0.6*pi/2;x10=0.9*pi/2; %10���źŵ������
doa(1)=x1;doa(2)=x2;doa(3)=x3;doa(4)=x4;doa(5)=x5;doa(6)=x6;doa(7)=x7;doa(8)=x8;doa(9)=x9;doa(10)=x10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�˴�����������Ӧ����
for jj=1:Num_signal

    for ii=1:array
    b(ii)=exp(-j*2*pi*(ii-1)*d*sin(doa(jj))/l) ;
    end 
    
B(:,jj)=b/(b'*b)^0.5;
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %�ô˴��������źż����������Ѿ�����
xxx=randint(Num_signal,Len);
xxx=2*(xxx-0.5);

Noise1=randn(array,Len)+j*randn(array,Len);
%save Noise1.mat Noise1
%save xxx.mat xxx
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%�����źż�����
% load Noise1.mat
% load xxx.mat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�ڼ��ֲ�ͬ��������¼���������
for SNR=-6:2:18
   %SNR=30;
%diBi=1/(2*SNR)^0.5;
%SNR=10;
diBi=1/(2*10^(SNR/10))^0.5;
Noise1=Noise1*diBi;%%%%���������

x_train=xxx(:,(1:Len_train));
N_train=Noise1(:,(1:Len_train));

x=B*x_train+N_train; %������
x1=B*xxx+Noise1; %������
R=x*x';
W=inv(R)*B*inv((B'*inv(R)*B))*e;

y=W'*x1;
output=real(y);
d_output=output;
d_output(find(d_output>=0))=1;
d_output(find(d_output<0))=-1;
d=xxx(1,:);
%error=norm(d-d_output,1)/2;

error=length(find(d-d_output))
pe=error/Len;
Plot_Pe=[Plot_Pe pe];
 
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

semilogy(Plot_SNR,Plot_Pe,'m*:');
axis([-6 18 10^(-5) 1 ]);
xlabel('SNR (dB)')
ylabel('BER');
%s=sprintf('BER versus SNR in the AWGN channel');
%title(s);
