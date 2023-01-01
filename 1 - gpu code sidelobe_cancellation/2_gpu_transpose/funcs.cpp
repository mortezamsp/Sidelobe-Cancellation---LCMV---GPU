
#include "matrix.h"
//using namespace std;

/**
 * returns a matrix with size cols x rows with ones as values
 */
matrix Ones(const int rows, const int cols)
{
  matrix res = matrix(rows, cols);

  for (int r = 1; r <= rows; r++)
  {
    for (int c = 1; c <= cols; c++)
    {
      res(r, c) = 1;
    }
  }
  return res;
};

/**
 * returns a matrix with size cols x rows with zeros as values
 */
matrix Zeros(const int rows, const int cols)
{
  return matrix(rows, cols);
};

