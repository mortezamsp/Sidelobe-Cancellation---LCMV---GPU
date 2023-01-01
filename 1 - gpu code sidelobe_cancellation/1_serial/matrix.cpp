
using namespace std;
#include "matrix.h"

void Swap(double& a, double& b)
{
  double temp = a;
  a = b;
  b = temp;
}

matrix::matrix()
  {
    //printf("Executing constructor matrix() ...\n");
    // create a matrix object without content
    p = NULL;
    rows = 0;
    cols = 0;
  }
matrix::matrix(const int row_count, const int column_count)
  {
    // create a matrix object with given number of rows and columns
    p = NULL;

    if (row_count > 0 && column_count > 0)
    {
      rows = row_count;
      cols = column_count;

      p = new double*[rows];
      for (int r = 0; r < rows; r++)
      {
        p[r] = new double[cols];

        // initially fill in zeros for all values in the matrix;
        for (int c = 0; c < cols; c++)
        {
          p[r][c] = 0;
        }
      }
	
	  matrixSpaceHasBeenAllocated = true;
    }
  }
matrix::matrix(const int row_count, const int column_count, const char *fillingtype, double value)
  {
    // create a matrix object with given number of rows and columns
    p = NULL;
	srand(rows);
    if (row_count > 0 && column_count > 0)
    {
      rows = row_count;
      cols = column_count;

      p = new double*[rows];
      for (int r = 0; r < rows; r++)
      {
        p[r] = new double[cols];

        // initially fill in zeros for all values in the matrix;
        for (int c = 0; c < cols; c++)
        {
			if(strcmp(fillingtype,"rand") == 0)
			{
				p[r][c] = rand() / RAND_MAX;
			}
			else if(strcmp(fillingtype,"num") == 0)
			{
				p[r][c] = rand() / value;
			}
			else
			{
				p[r][c] = 0;
			}
        }
      }
    }
	matrixSpaceHasBeenAllocated = true;
  }
matrix::matrix(const matrix& a)
  {
    rows = a.rows;
    cols = a.cols;
    p = new double*[a.rows];
    for (int r = 0; r < a.rows; r++)
    {
      p[r] = new double[a.cols];

      // copy the values from the matrix a
      for (int c = 0; c < a.cols; c++)
      {
        p[r][c] = a.p[r][c];
      }
    }
	matrixSpaceHasBeenAllocated = true;
  }

double matrix::sum()
  {
	  double s = 0;
	  for(int i=0; i<rows; i++)
		  for(int j=0; j<cols; j++)
			  s += p[i][j];
	  return s;
  }
double& matrix::operator()(const int r, const int c)
  {
    if (p != NULL && r >= 0 && r < rows && c >= 0 && c < cols)
    {
      return p[r][c];
    }
    else
    {
      throw MatrixException("Subscript out of range");
    }
  }
double matrix::get(const int r, const int c) const
  {
    if (p != NULL && r >= 0 && r < rows && c >= 0 && c < cols)
    {
      return p[r][c];
    }
    else
    {
      throw MatrixException("Subscript out of range");
    }
  }
matrix& matrix::operator= (const matrix& a)
  {
    rows = a.rows;
    cols = a.cols;
	delete []p;
    p = new double*[a.rows];
    for (int r = 0; r < a.rows; r++)
    {
      p[r] = new double[a.cols];

      // copy the values from the matrix a
      for (int c = 0; c < a.cols; c++)
      {
        p[r][c] = a.p[r][c];
      }
    }
	matrixSpaceHasBeenAllocated = true;
    return *this;
  }
matrix& matrix::Add(const double v)
  {
    for (int r = 0; r < rows; r++)
    {
      for (int c = 0; c < cols; c++)
      {
        p[r][c] += v;
      }
    }
     return *this;
  }
matrix& matrix::Subtract(const double v)
  {
    return Add(-v);
  }
matrix& matrix::Multiply(const double v)
  {
    for (int r = 0; r < rows; r++)
    {
      for (int c = 0; c < cols; c++)
      {
        p[r][c] *= v;
      }
    }
     return *this;
  }
matrix& matrix::Divide(const double v)
  {
     return Multiply(1/v);
  }
matrix matrix::Minor(const int row, const int col) const
  {
    matrix res;
    if (row >= 0 && row < rows && col >= 0 && col < cols)
    {
      res = matrix(rows, cols);

      // copy the content of the matrix to the minor, except the selected
      for (int r = 1; r <= (rows - (row >= rows)); r++)
      {
        for (int c = 1; c <= (cols - (col >= cols)); c++)
        {
          res(r - (r > row), c - (c > col)) = p[r-1][c-1];
        }
      }
    }
    else
    {
      throw MatrixException("Index for minor out of range");
    }

    return res;
  }
int matrix::Size(const int i) const
  {
    if (i == 1)
    {
      return rows;
    }
    else if (i == 2)
    {
      return cols;
    }
    return 0;
  }
int matrix::GetRows() const
  {
    return rows;
  }
int matrix::GetCols() const
  {
    return cols;
  }
matrix::~matrix()
  {
    // clean up allocated memory
//    for (int r = 0; r < rows; r++)
//    {
//      delete p[r];
//    }
    delete p;
    p = NULL;
  }
matrix matrix::Inv()
{
  matrix res;
  double d = 0;    // value of the determinant

  d = Det();
  if (rows == cols && d != 0)
  {
    // this is a square matrix
    if (rows == 1)
    {
      // this is a 1 x 1 matrix
      res = matrix(rows, cols);
      res(0, 0) = 1 / get(1, 1);
    }
    else if (rows == 2)
    {
      // this is a 2 x 2 matrix
      res = matrix(rows, cols);
      res(0, 0) = get(2, 2);
      res(0, 1) = -get(1, 2);
      res(1, 0) = -get(2, 1);
      res(1, 1) = get(1, 1);
      res = (1/d) * res;
    }
    else
    {
      // this is a matrix of 3 x 3 or larger
      // calculate inverse using gauss-jordan elimination
      //   http://mathworld.wolfram.com/matrixInverse.html
      //   http://math.uww.edu/~mcfarlat/inverse.htm
      res = Diag(rows);   // a diagonal matrix with ones at the diagonal
	  matrix ai = *this;    // make a copy of matrix a

      for (int c = 0; c < cols; c++)
      {
        // element (c, c) should be non zero. if not, swap content
        // of lower rows
        int r;
        for (r = c; r < rows && ai(r, c) == 0; r++)
        {
        }
        if (r != c)
        {
          // swap rows
          for (int s = 0; s < cols; s++)
          {
            Swap(ai(c, s), ai(r, s));
            Swap(res(c, s), res(r, s));
          }
        }

        // eliminate non-zero values on the other rows at column c
        for (int r = 0; r < rows; r++)
        {
          if(r != c)
          {
            // eleminate value at column c and row r
            if (ai(r, c) != 0)
            {
              double f = - ai(r, c) / ai(c, c);

              // add (f * row c) to row r to eleminate the value
              // at column c
              for (int s = 0; s < cols; s++)
              {
                ai(r, s) += f * ai(c, s);
                res(r, s) += f * res(c, s);
              }
            }
          }
          else
          {
            // make value at (c, c) one,
            // divide each value on row r with the value at ai(c,c)
            double f = ai(c, c);
            for (int s = 0; s < cols; s++)
            {
              ai(r, s) /= f;
              res(r, s) /= f;
            }
          }
        }
      }
    }
  }
  else
  {
    if (rows == cols)
    {
      //throw MatrixException("matrix must be square");
	#ifdef win32
		return matrix(cols, rows, "num", LONG_MAX);
	#else
		return matrix(cols, rows, "num", 99999999);
	#endif
    }
    else
    {
      throw MatrixException("Determinant of matrix is zero");
    }
  }
  return res;
}
#include<sys/time.h>
matrix matrix::transpose()
{
struct timeval s,e;
gettimeofday(&s,NULL);
	matrix res(cols, rows);
	for(int i=0; i<rows; i++)
		for(int j=0; j<cols; j++)
			res(j, i) = get(i, j);
gettimeofday(&e,NULL);
printf("\nexecuting 'transpose' function took %f milli-secconds.",(e.tv_sec-s.tv_sec)*1000+(e.tv_usec-s.tv_usec)/1000.0);
	return res;
}
double matrix::Det()
{
	//using sarrus rule :

  double sum1 = 0;
  for(int j=0; j<cols; j++)
  {
	  double ts=1;
	  for(int i=0; i<rows; i++)
		  ts *= p[i][(i+j) % cols];
	  sum1 += ts;
  }

  double sum2 = 0;
  for(int j=cols-1; j>0; j--)
  {
	  double ts=1;
	  for(int i=0; i<rows; i++)
		  ts *= p[i][(j-i+rows-1) % cols];
	  sum2 += ts;
  }

  return sum1 - sum2;
}
matrix matrix::Diag(const int n)
{
  matrix res = matrix(n, n);
  for (int i = 0; i < n; i++)
  {
    res(i, i) = 1;
  }
  return res;
}
matrix matrix::Diag()
{
  matrix res;
  if (cols == 1)
  {
    // the given matrix is a vector n x 1
    res = matrix(rows, rows);

    // copy the values of the vector to the matrix
    for (int r=1; r <= rows; r++)
    {
      res(r, r) = get(r, 1);
    }
  }
  else if (rows == 1)
  {
    // the given matrix is a vector 1 x n
    res = matrix(cols, cols);

    // copy the values of the vector to the matrix
    for (int c=1; c <= cols; c++)
    {
      res(c, c) = get(1, c);
    }
  }
  else
  {
    throw MatrixException("Parameter for diag must be a vector");
  }
  return res;
}
