#include <iostream>
#ifdef win32
	#include <time.h>
#else
	#include<sys/time.h>
#endif

using namespace std;

#include "matrix.h"
//#include "funcs.cpp"
#include "vars.h"

void init_vars();


int main()
{
    init_vars();

#ifdef WIN32
	clock_t t = clock();
#else
	struct timeval start, end;
	long mtime, seconds, useconds;
	gettimeofday(&start, NULL);
#endif

    for(int jj=0; jj<Num_signal; jj++)
    {
        for(int ii=0; ii<array; ii++)
            b(ii, 0) = pow(10, (-(j+1)*2*pi*ii*d*sin(doa.get(jj, 0))/l));
        //B(:,jj)=b/(b'*b)^0.5;
        double bsum = pow(b.sum(), 0.5);
        for(int ii=0; ii<array; ii++)
            B(ii, jj) = b.get(ii, 0) / bsum;
    }

    //xxx=randint(Num_signal,Len);
    //xxx=2*(xxx-0.5);
    matrix xxx(Num_signal, Len, "rand", 0);
    xxx = (xxx - 0.5) * 2.0;
	matrix d(1, xxx.GetCols());

    //Noise1=randn(array,Len)+j*randn(array,Len);
    matrix randn1(array, Len, "rand", 0), randn2(array, Len, "rand", 0);
    matrix Noise1(array, Len);
    Noise1 = randn1 + randn2 * j;
	randn1.~matrix();
	randn2.~matrix();


    matrix *errs = new matrix[13];
    for(int SNR=-6; SNR<=18; SNR+=2)
        //SNR=30;
    {
        //diBi=1/(2*SNR)^0.5; if SNR=10;
        double diBi = 1 / pow(2*pow(10, SNR/10), 0.5);

        Noise1 = Noise1 * diBi;

        //x_train=xxx(:,(1:Len_train));
        for(int i=0; i<Num_signal; i++)
            for(int ii=0; ii<Len_train; ii++)
                x_train(i, ii) = xxx.get(i, ii);

        //N_train=Noise1(:,(1:Len_train));
        for(int i=0; i<array; i++)
            for(int ii=0; ii<Len_train; ii++)
                N_train(i, ii) = Noise1.get(i, ii);

        x = B * x_train;
		x += N_train;
        x1 = B * xxx;
		x1 += Noise1;
		R = x * x.transpose();
		R = R.Inv();

        //W=inv(R)*B*inv((B'*inv(R)*B))*e;
		W = R * B;
		W = W * (B.transpose() * R * B).Inv();
		W = W * e;

        //y=W'*x1;
        y = W.transpose() * x1;

        //output=real(y);
        //d_output=output;
        //d_output(find(d_output>=0))=1;
        //d_output(find(d_output<0))=-1;
		if(! d_output.matrixSpaceHasBeenAllocated)
		{
			d_output = matrix(y.GetRows(), y.GetCols());
			//printf("\n sapce allocated to matrix d_output.");
		}
        for(int i=0; i<y.GetRows(); i++)
        {
            for(int j=0; j<y.GetCols(); j++)
            {
                if(y.get(i, j) >= 0)
                    d_output(i, j) = 1;
                else
                    d_output(i, j) = -1;
            }
        }

        //d=xxx(1,:);
        for(int i=0; i<xxx.GetRows(); i++)
            d(0, i) = xxx(0, i);
        //error=norm(d-d_output,1)/2;

        //error=length(find(d-d_output))
        //pe=error/Len;
        //Plot_Pe=[Plot_Pe pe];
        errs[(SNR+6)/2] = d - d_output;


		//printf("\nSNR[%d]:\tYr=%d\tYc=%d\tXXXc=%d",SNR,y.GetRows(),y.GetCols(),xxx.GetCols());
		printf("\nSNR[%d]",SNR);
    }

#ifdef WIN32
	t = clock() - t;
	printf("\n\nTotal Running Time : %f (secconds)", ((float)t)/CLOCKS_PER_SEC);
	printf("\npress enter to exit");
    	getchar();

#else
	gettimeofday(&end, NULL);
	seconds  = end.tv_sec  - start.tv_sec;
    	useconds = end.tv_usec - start.tv_usec;
    	mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    	printf("\n\nTotal Running Time : %ld (milli secconds)\n", mtime);
#endif

    return 0;
}


void init_vars()
{
    e(0,0)=1;
    double x1=-0.6*pi/2, x2=-pi/3, x3=-pi/4 ,x4=-pi/6 ,x5=0.9*pi ,x6=pi/6 ,x7=pi/4 ,x8=pi/3 ,x9=0.6*pi/2 ,x10=0.9*pi/2;
    doa(0,0)=x1;
    doa(1,0)=x2;
    doa(2,0)=x3;
    doa(3,0)=x4;
    doa(4,0)=x5;
    doa(5,0)=x6;
    doa(6,0)=x7;
    doa(7,0)=x8;
    doa(8,0)=x9;
    doa(9,0)=x10;
}
