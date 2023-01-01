
#ifndef MATRIXEXCEPTION_H_
#define MATRIXEXCEPTION_H_

#include <stdio.h>
#include <stdlib.h>


#define PAUSE {printf("Press \"Enter\" to continue\n"); fflush(stdin); getchar(); fflush(stdin);}

/*
 * a simple MatrixException class
 * you can create an exeption by entering
 *   throw MatrixException("...Error description...");
 * and get the error message from the data msg for displaying:
 *   err.msg
 */
class MatrixException
{
public:
  const char* msg;
  MatrixException(const char* arg)
   : msg(arg)
  {
      PAUSE
      exit(0);
  }
};

#endif
