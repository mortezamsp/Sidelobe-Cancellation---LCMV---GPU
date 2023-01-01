
#include "MatrixException.cpp"

#ifndef MATRIX_H_
#define MATRIX_H_

#include <string.h>
#include <math.h>


class matrix
{
public:
	  // constructor
	bool matrixSpaceHasBeenAllocated;
	  matrix();
	  matrix(const int row_count, const int column_count);
	  matrix(const int row_count, const int column_count, const char *fillingtype, double value);
	  matrix(const matrix& a);
	  matrix(const int row_count, const int column_count, double **array);

	  // destructor
	  ~matrix();

	  // index operator. You can use this class like mymatrix(col, row)
	  // the indexes are one-based, not zero based.
	  double& operator()(const int r, const int c);

	  // index operator. You can use this class like mymatrix.get(col, row)
	  // the indexes are one-based, not zero based.
	  // use this function get if you want to read from a const matrix
	  double get(const int r, const int c) const;

	  // assignment operator
	  matrix& operator= (const matrix& a);

	  // add a double value (elements wise)
	  matrix& Add(const double v);

	  // subtract a double value (elements wise)
	  matrix& Subtract(const double v);

	  // multiply a double value (elements wise)
	  matrix& Multiply(const double v);

	  // divide a double value (elements wise)
	  matrix& Divide(const double v);

	  // addition of matrix with matrix
	  friend matrix operator+(const matrix& a, const matrix& b)
	  {
		// check if the dimensions match
		if (a.rows == b.rows && a.cols == b.cols)
		{
		  matrix res(a.rows, a.cols);

		  for (int r = 0; r < a.rows; r++)
		  {
			for (int c = 0; c < a.cols; c++)
			{
			  res.p[r][c] = a.p[r][c] + b.p[r][c];
			}
		  }
		  return res;
		}
		else
		{
		  // give an error
		  throw MatrixException("Dimensions does not match");
		}

		// return an empty matrix (this never happens but just for safety)
		return matrix();
	  }
	  friend void operator+=(const matrix& a, const matrix& b)
	  {
		// check if the dimensions match
		if (a.rows == b.rows && a.cols == b.cols)
		{
		  for (int r = 0; r < a.rows; r++)
		  {
			for (int c = 0; c < a.cols; c++)
			{
			  a.p[r][c] += b.p[r][c];
			}
		  }
		}
		else
		{
		  // give an error
		  throw MatrixException("Dimensions does not match");
		}
	  }
	  friend matrix operator+ (const matrix& a, const double b)
	  {
		matrix res = a;
		res.Add(b);
		return res;
	  }
	  // addition of double with matrix
	  friend matrix operator+ (const double b, const matrix& a)
	  {
		matrix res = a;
		res.Add(b);
		return res;
	  }

	  // subtraction of matrix with matrix
	  friend matrix operator- (const matrix& a, const matrix& b)
	  {
		// check if the dimensions match
		if (a.rows == b.rows && a.cols == b.cols)
		{
		  matrix res(a.rows, a.cols);

		  for (int r = 0; r < a.rows; r++)
		  {
			for (int c = 0; c < a.cols; c++)
			{
			  res.p[r][c] = a.p[r][c] - b.p[r][c];
			}
		  }
		  return res;
		}
		else
		{
		  // give an error
		  throw MatrixException("Dimensions does not match");
		}

		// return an empty matrix (this never happens but just for safety)
		return matrix();
	  }
	  friend void operator-=(const matrix& a, const matrix& b)
	  {
		// check if the dimensions match
		if (a.rows == b.rows && a.cols == b.cols)
		{
		  for (int r = 0; r < a.rows; r++)
		  {
			for (int c = 0; c < a.cols; c++)
			{
			  a.p[r][c] -= b.p[r][c];
			}
		  }
		}
		else
		{
		  // give an error
		  throw MatrixException("Dimensions does not match");
		}
	  }

	  // subtraction of matrix with double
	  friend matrix operator- (const matrix& a, const double b)
	  {
		matrix res = a;
		res.Subtract(b);
		return res;
	  }
	  // subtraction of double with matrix
	  friend matrix operator- (const double b, const matrix& a)
	  {
		matrix res = -a;
		res.Add(b);
		return res;
	  }

	// operator unary minus
	friend matrix operator- (const matrix& a)
	  {
		matrix res(a.rows, a.cols);

		for (int r = 0; r < a.rows; r++)
		{
		  for (int c = 0; c < a.cols; c++)
		  {
			res.p[r][c] = -a.p[r][c];
		  }
		}

		return res;
	  }

	// operator multiplication
	friend matrix operator* (const matrix& a, const matrix& b)
	  {
		// check if the dimensions match
		if (a.cols == b.rows)
		{
		  matrix res(a.rows, b.cols);

		  for (int r = 0; r < a.rows; r++)
		  {
			for (int c_res = 0; c_res < b.cols; c_res++)
			{
			  for (int c = 0; c < a.cols; c++)
			  {
				res.p[r][c_res] += a.p[r][c] * b.p[c][c_res];
			  }
			}
		  }
		  return res;
		}
		else
		{
			// give an error
			printf("Dimensions does not match { matrix a[%d,%d] , matrix b[%d,%d] }", a.rows,a.cols,b.rows,b.cols);
			throw MatrixException("\nDimensions ERR occured.");
		}

		// return an empty matrix (this never happens but just for safety)
		return matrix();
	  }
	// multiplication of matrix with double
	friend matrix operator* (const matrix& a, const double b)
	  {
		matrix res = a;
		res.Multiply(b);
		return res;
	  }
	// multiplication of double with matrix
	friend matrix operator* (const double b, const matrix& a)
	  {
		matrix res = a;
		res.Multiply(b);
		return res;
	  }

	// division of matrix with double
	friend matrix operator/ (const matrix& a, const double b)
	  {
		matrix res = a;
		res.Divide(b);
		return res;
	  }
	/**
	   * returns the minor from the given matrix where
	   * the selected row and column are removed
	   */
	matrix Minor(const int row, const int col) const;
  
	matrix Inv();
	//CUDA_CALLABLE_MEMBER 
	matrix& getInstance();
	matrix& transpose();
	//__global__ void CUDA_transpose(double *, double *);
	double Det();
	/**
	 * returns a diagonal matrix with size n x n with ones at the diagonal
	 * @param  v a vector
	 * @return a diagonal matrix with ones on the diagonal
	 */
	matrix Diag(const int n);
	matrix Diag();

  /*
   * returns the size of the i-th dimension of the matrix.
   * i.e. for i=1 the function returns the number of rows,
   * and for i=2 the function returns the number of columns
   * else the function returns 0
   */
  int Size(const int i) const;

  // some of matrix
  double sum();

  // returns the number of rows
  int GetRows() const;

  // returns the number of columns
  int GetCols() const;

private:
	  int rows;
	  int cols;
	  double** p;     // pointer to a matrix with doubles
};

#endif
