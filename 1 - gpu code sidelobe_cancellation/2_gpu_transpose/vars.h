
#include "matrix.h"

int array = 25;
int Num_signal = 10;

//#define name=ZEROS(up1,up2) doulbe**handle##name=new float*[handle##up1];for(inti=0;i<up1;i++)handle##name[i]=new float[handle##up1];
matrix b(array, 1);
matrix B(array, Num_signal);
matrix doa(Num_signal, 1);
matrix AA(1, array);
float l = 0.06;
float d = 0.5 * l;
matrix e(10, 1);
int Len_train = 100;
int Len = 100000;

matrix x_train(Num_signal,Len_train);//signal for training
matrix N_train(array,Len_train);//noise for training

matrix xxx(Num_signal,Len);//signal
matrix Noise1(array,Len);//noise

//float Plot_SNR = -6:2:18;
//Plot_Pe=[];

matrix W(array,1);

float pi = 3.14159265359;

float j = 1;



// these matrixes are produced incide the main's LOOP,
//	so inorder to refuse 'HEAP allocation' problems, i define them heare for just one time.
matrix x(1,1);
matrix R(1,1);
matrix x1(1,1);
matrix y(1,1);
matrix d_output(0,0);
